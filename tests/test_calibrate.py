import json
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
from scipy.io import wavfile

import config
from calibration import (
    CalibrationFileError,
    apply_calibration,
    build_calibration,
    compute_offset,
    evaluate_background,
    is_calibrated,
    load_calibration,
    make_reading,
    save_mic_calibration,
    spread_warning,
    warn_uncalibrated,
)
from capture import FileSource
from level import to_estimated_dba
from tools.calibrate import (
    LevelMonitor,
    ask_meter_value,
    parse_meter_value,
    report_calibration,
    run_calibration,
)

SAMPLE_RATE = 48000


@pytest.fixture(autouse=True)
def restore_offsets() -> Iterator[None]:
    saved = dict(config.CALIBRATION_OFFSET_DB)
    yield
    config.CALIBRATION_OFFSET_DB.clear()
    config.CALIBRATION_OFFSET_DB.update(saved)


def make_calibration(offsets: list[float], background_dbfs_a: float = -70.0):
    readings = [
        make_reading(60.0 + index, 60.0 + index - offset)
        for index, offset in enumerate(offsets)
    ]
    return build_calibration(
        readings, background_dbfs_a, 48000, "18", "2026-10-10T14:00:00+09:00"
    )


# --- 계산과 검증 ---


def test_offset_is_meter_minus_program() -> None:
    assert compute_offset(62.0, -31.4) == pytest.approx(93.4)


def test_calibration_averages_levels_and_converts_background() -> None:
    calibration = make_calibration([93.0, 94.0, 95.0], background_dbfs_a=-70.0)
    assert calibration.offset_db == pytest.approx(94.0)
    assert calibration.background_leq_db == pytest.approx(24.0)
    assert len(calibration.levels) == 3


def test_spread_within_limit_is_silent() -> None:
    readings = [make_reading(60.0, -33.0), make_reading(70.0, -22.5)]  # 93.0, 92.5
    assert spread_warning(readings, max_spread_db=2.0) is None


def test_spread_over_limit_warns() -> None:
    readings = [make_reading(60.0, -33.0), make_reading(70.0, -26.0)]  # 93, 96
    message = spread_warning(readings, max_spread_db=2.0)
    assert message is not None and "AGC" in message


def test_single_level_has_no_spread_warning() -> None:
    assert spread_warning([make_reading(60.0, -33.0)], max_spread_db=2.0) is None


@pytest.mark.parametrize(
    "background, expected_keywords, expected_gate",
    [
        (15.0, [], 18.0),  # 배경 + 3
        (24.0, ["건너뛰기가 일어나지 않음"], 24.0),  # min(27, 34 − 10)
        (30.0, ["건너뛰기가 일어나지 않음", "오탐 위험"], 24.0),
    ],
)
def test_background_checks_and_recommended_gate(
    background: float, expected_keywords: list[str], expected_gate: float
) -> None:
    warnings, recommended = evaluate_background(
        background,
        gate_db=24.0,
        min_limit_db=34.0,
        margin_db=10.0,
        near_limit_db=5.0,
        headroom_db=3.0,
    )
    assert len(warnings) == len(expected_keywords)
    for keyword, warning in zip(expected_keywords, warnings, strict=True):
        assert keyword in warning
    assert recommended == pytest.approx(expected_gate)


def test_report_prints_recommendation_without_changing_gate(
    capsys: pytest.CaptureFixture,
) -> None:
    gate_before = config.SKIP_CLASSIFY_BELOW_DB
    warnings = report_calibration(
        "거실", make_calibration([93.0], background_dbfs_a=-63.0)
    )
    output = capsys.readouterr().out
    assert "권장 게이트: 24.0 dB(A)" in output  # 배경 30 → min(33, 24)
    assert any("오탐 위험" in warning for warning in warnings)
    assert config.SKIP_CLASSIFY_BELOW_DB == gate_before


# --- 파일 ---


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    calibration = make_calibration([93.0, 94.0])
    save_mic_calibration(path, "거실", calibration)
    assert load_calibration(path) == {"거실": calibration}
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored["거실"]) >= {
        "offset_db",
        "meter_leq",
        "program_leq",
        "levels",
        "background_leq_db",
        "sample_rate",
        "device",
        "measured_at",
    }


def test_saving_another_room_keeps_existing(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save_mic_calibration(path, "거실", make_calibration([93.0]))
    save_mic_calibration(path, "안방", make_calibration([91.0]))
    save_mic_calibration(path, "거실", make_calibration([95.0]))
    loaded = load_calibration(path)
    assert loaded["거실"].offset_db == pytest.approx(95.0)
    assert loaded["안방"].offset_db == pytest.approx(91.0)


def test_missing_file_means_no_calibration(tmp_path: Path) -> None:
    assert load_calibration(tmp_path / "none.json") == {}


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"거실": {"offset_db": 93.4,}}', "JSON 형식 오류"),
        ("[1, 2]", "최상위"),
        ('{"거실": {"meter_leq": 60}}', "'거실' 보정값"),
        ('{"거실": {"offset_db": "많이"}}', "'거실' 보정값"),
    ],
    ids=["broken_json", "not_object", "missing_offset", "offset_not_number"],
)
def test_bad_file_gives_clear_error(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CalibrationFileError, match=message):
        load_calibration(path)


def test_broken_file_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CalibrationFileError):
        save_mic_calibration(path, "거실", make_calibration([93.0]))
    assert path.read_text(encoding="utf-8") == "{broken"


def test_uncalibrated_room_uses_default_offset_and_is_reported() -> None:
    apply_calibration({"거실": make_calibration([93.0])})
    assert to_estimated_dba(-40.0, "거실") == pytest.approx(53.0)
    assert to_estimated_dba(-40.0, "안방") == pytest.approx(
        -40.0 + config.DEFAULT_CALIBRATION_OFFSET_DB
    )
    assert is_calibrated("거실") and not is_calibrated("안방")
    assert warn_uncalibrated(["거실", "안방"]) == ["안방"]


# --- 입력 ---


def test_parse_meter_value_accepts_units() -> None:
    assert parse_meter_value(" 62.5 dB(A) ") == 62.5


@pytest.mark.parametrize("text", ["", "abc", "-3", "200"])
def test_parse_meter_value_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_meter_value(text)


def test_ask_meter_value_retries_until_valid() -> None:
    answers = iter(["", "열두", "61.0"])
    assert ask_meter_value("> ", lambda _: next(answers)) == 61.0


# --- 측정 절차 (파일을 마이크처럼) ---


def test_run_calibration_measures_with_level_meter(tmp_path: Path) -> None:
    # 1kHz, 진폭 0.1 → 약 −23 dBFS(A). 소음계 70 → 오프셋 약 93.
    time_axis = numpy.arange(6 * SAMPLE_RATE) / SAMPLE_RATE
    path = tmp_path / "tone.wav"
    wavfile.write(
        path,
        SAMPLE_RATE,
        (0.1 * numpy.sin(2 * numpy.pi * 1000 * time_axis)).astype(numpy.float32),
    )
    source = FileSource("거실", path, end_behavior="stop")
    monitor = LevelMonitor([source], realtime_pacing=True)
    monitor.start()
    answers = iter(["", "", "70"])
    try:
        calibration = run_calibration(
            monitor, "거실", 1, 1.0, SAMPLE_RATE, "file", lambda _: next(answers)
        )
    finally:
        monitor.stop_event.set()
        monitor.join(5.0)
    assert calibration.program_leq == pytest.approx(-23.0, abs=0.3)
    assert calibration.offset_db == pytest.approx(93.0, abs=0.3)
    assert calibration.background_leq_db == pytest.approx(70.0, abs=0.3)
