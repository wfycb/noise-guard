"""이벤트 녹음 테스트 (합성 신호). tick마다 오디오 값을 tick 번호로 채워 구간을 샘플 단위로 확인한다."""

import json
from pathlib import Path

import numpy
import pytest
from scipy.io import wavfile

from event_recorder import EventRecorder
from models import Alert, Category, Frame, MicMeasurement

BASE = 1_000_000.0
RATES = {"거실": 1000, "안방": 800}


def tick_audio(tick: int, rates: dict[str, int], skip: set[str] = frozenset()) -> dict:
    """tick k(끝 시각 BASE + k)의 1초 오디오. 값은 k / 100이라 어느 tick인지 알 수 있다."""
    return {
        room: numpy.full(rate, tick / 100.0, dtype=numpy.float32)
        for room, rate in rates.items()
        if room not in skip
    }


def make_alert(tick: int, rule: str = "R1", level: str = "caution") -> Alert:
    return Alert(
        level, rule, Category.IMPACT, "거실", 60.0, 1, BASE + tick, f"{rule} m"
    )


def make_frame(tick: int) -> tuple[Frame, dict[str, MicMeasurement]]:
    probs = {category: 0.0 for category in Category}
    probs[Category.IMPACT] = 0.9
    measurement = MicMeasurement(60.0, 65.0, probs, "Walk, footsteps")
    frame = Frame(BASE + tick, "거실", 60.0, 65.0, probs, "Walk, footsteps")
    return frame, {"거실": measurement, "안방": measurement}


def run_ticks(
    recorder: EventRecorder,
    ticks: range,
    alert_ticks: dict[int, list[Alert]],
    rates: dict[str, int] = RATES,
    missing: dict[int, set[str]] | None = None,
) -> None:
    recorder.start_stream(rates)
    for tick in ticks:
        frame, measurements = make_frame(tick)
        recorder.on_tick(
            BASE + tick,
            tick_audio(tick, rates, (missing or {}).get(tick, set())),
            frame,
            measurements,
            alert_ticks.get(tick, []),
        )
    recorder.close()


def read_event(folder: Path, room: str) -> tuple[int, numpy.ndarray]:
    sample_rate, samples = wavfile.read(folder / f"{room}.wav")
    return sample_rate, samples.astype(numpy.float64) / 32768.0


def tick_values(samples: numpy.ndarray, rate: int) -> list[float]:
    """1초마다 대표값(tick 번호/100)을 읽는다."""
    return [
        round(float(samples[index * rate]) * 100)
        for index in range(len(samples) // rate)
    ]


def make_recorder(tmp_path: Path, **overrides: float) -> EventRecorder:
    settings = {"pre_sec": 2.0, "post_sec": 3.0, "max_sec": 30.0, "max_total_mb": 100.0}
    settings.update(overrides)
    return EventRecorder(
        tmp_path / "events", context={"model": "test-model"}, **settings
    )


def test_pre_two_and_post_three_seconds_exact_samples(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(recorder, range(1, 12), {5: [make_alert(5)]})
    [folder] = recorder.saved_folders
    for room, rate in RATES.items():
        sample_rate, samples = read_event(folder, room)
        assert sample_rate == rate
        assert len(samples) == 5 * rate
        # 알림 시각 BASE+5 기준 앞 2초 = tick 4, 5 / 뒤 3초 = tick 6, 7, 8
        assert tick_values(samples, rate) == [4, 5, 6, 7, 8]


def test_new_alert_extends_end_only(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(recorder, range(1, 15), {5: [make_alert(5)], 7: [make_alert(7)]})
    [folder] = recorder.saved_folders
    _, samples = read_event(folder, "거실")
    assert tick_values(samples, 1000) == [4, 5, 6, 7, 8, 9, 10]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert [alert["timestamp"] - BASE for alert in meta["alerts"]] == [5, 7]


def test_extension_is_capped_at_max_length(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path, max_sec=6.0)
    alerts = {tick: [make_alert(tick)] for tick in (5, 6, 7, 8, 9)}
    run_ticks(recorder, range(1, 20), alerts)
    folders = recorder.saved_folders
    _, samples = read_event(folders[0], "거실")
    assert len(samples) == 6 * 1000
    assert tick_values(samples, 1000) == [4, 5, 6, 7, 8, 9]


def test_early_alert_pads_missing_pre_audio(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(recorder, range(1, 8), {1: [make_alert(1)]})
    [folder] = recorder.saved_folders
    _, samples = read_event(folder, "거실")
    assert len(samples) == 5000
    assert tick_values(samples, 1000) == [0, 1, 2, 3, 4]  # 앞 1초는 무음으로 채움
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["pre_padded_sec"] == pytest.approx(1.0)


def test_missing_mic_tick_is_filled_with_silence(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(recorder, range(1, 12), {5: [make_alert(5)]}, missing={7: {"안방"}})
    [folder] = recorder.saved_folders
    _, living = read_event(folder, "거실")
    _, bedroom = read_event(folder, "안방")
    assert tick_values(living, 1000) == [4, 5, 6, 7, 8]
    assert tick_values(bedroom, 800) == [
        4,
        5,
        6,
        0,
        8,
    ]  # 모든 마이크 저장, 빠진 tick은 무음


def test_stream_end_saves_truncated_recording(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(recorder, range(1, 7), {5: [make_alert(5)]})
    [folder] = recorder.saved_folders
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["truncated"] is True
    _, samples = read_event(folder, "거실")
    assert tick_values(samples, 1000) == [4, 5, 6]


def test_meta_has_required_fields(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    run_ticks(
        recorder,
        range(1, 12),
        {5: [make_alert(5), make_alert(5, rule="R3", level="warning")]},
    )
    [folder] = recorder.saved_folders
    assert folder.name.endswith("_R1+R3")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    for key in (
        "first_alert_time",
        "pre_sec",
        "post_sec",
        "duration_sec",
        "representative_mic",
        "alerts",
        "frames",
        "sample_rates",
        "calibration",
        "model",
    ):
        assert key in meta, key
    assert meta["alerts"][0]["level"] == "warning"
    assert meta["alerts"][0]["rules"] == ["R1", "R3"]
    assert meta["representative_mic"] == "거실"
    assert meta["model"] == "test-model"
    assert set(meta["calibration"]) == set(RATES)
    frame_ticks = [frame["timestamp"] - BASE for frame in meta["frames"]]
    assert frame_ticks == [4, 5, 6, 7, 8]
    first = meta["frames"][0]["mics"]["거실"]
    assert set(first) == {"leq_db", "lmax_db", "category_probs", "top_label"}


def test_total_size_limit_deletes_oldest(tmp_path: Path) -> None:
    # 이벤트 하나 ≈ wav 18KB(5초 × (1000 + 800) × 2바이트) + meta.json 약 5KB.
    # 상한 60KB → 두 개까지는 남고, 세 번째 저장 때 첫 번째를 지운다.
    recorder = make_recorder(tmp_path, max_total_mb=0.06)
    alerts = {5: [make_alert(5)], 15: [make_alert(15)], 25: [make_alert(25)]}
    run_ticks(recorder, range(1, 32), alerts)
    remaining = sorted(path.name for path in (tmp_path / "events").iterdir())
    assert len(recorder.saved_folders) == 3
    assert recorder.deleted_folders == [recorder.saved_folders[0].name]
    assert remaining == [folder.name for folder in recorder.saved_folders[1:]]
