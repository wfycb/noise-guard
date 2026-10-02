"""ESC-50 시나리오 회귀 테스트 (CED 모델과 ESC-50 클립 필요, 느림).

실행: .venv\\Scripts\\python -m pytest -m slow
데모 모드, 주간 14:00 가상 시계, 대기 없는 고속 재생으로 main.run_sources를 그대로 돌린다.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from capture import FileSource
from decision import LOCAL_TIMEZONE, DecisionConfig, NoiseDecisionEngine
from tools.make_scenario import SCENARIOS, write_scenario

if TYPE_CHECKING:
    from classifier import CedClassifier
    from main import RunResult

pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ESC50_AUDIO_DIR = PROJECT_ROOT / "data" / "esc50" / "audio"
DAY_START = datetime(2026, 10, 2, 14, 0, tzinfo=LOCAL_TIMEZONE).timestamp()

# 시나리오 → 기대 알림 (규칙별 횟수). 보정 전 임시 오프셋(+100 dB) 기준이다.
EXPECTED_RULE_COUNTS: dict[str, dict[str, int]] = {
    # 발소리 이벤트 3개 → caution 3 + R2 1. 레벨이 기준보다 5~19 dB 커서 R3(데모 20초)도 매번 발령.
    "S1_footsteps_x3": {"R1": 3, "R2": 1, "R3": 3},
    "S2_water_only": {},
    # 알려진 한계: 물소리(약 -13 dBFS)가 발소리(약 -34 dBFS)를 가려 IMPACT가 임계값에 못 미친다.
    "S3_water_plus_footsteps": {},
    # 로직 검증용: 물소리를 20 dB 낮추면 두 쌍 중 한 쌍(따르는 물 + 발소리)만 IMPACT로 잡힌다.
    "S3b_quiet_water_plus_footsteps": {"R1": 1, "R3": 1},
    # 개·아기 울음 연속 75초 → R4가 첫 프레임부터 발령되고 쿨다운(데모 10초)마다 재발령.
    "S4_dog_and_baby": {"R4": 7},
    # 거실 발소리 → 안방 발소리. 카운트는 집 전체 기준이라 R1 두 번째는 2회.
    "S5_two_rooms": {"R1": 2, "R3": 2},
}


@pytest.fixture(scope="module")
def classifier() -> CedClassifier:
    # torch/transformers import는 수 초 걸리므로, slow 테스트를 실제로 돌릴 때만 한다.
    from classifier import CedClassifier

    return CedClassifier()


@pytest.fixture(scope="module")
def scenario_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not ESC50_AUDIO_DIR.is_dir():
        pytest.skip(f"ESC-50 클립이 없습니다: {ESC50_AUDIO_DIR}")
    output_dir = tmp_path_factory.mktemp("scenarios")
    for name, tracks in SCENARIOS.items():
        write_scenario(name, tracks, ESC50_AUDIO_DIR, output_dir, extra_gain_db=0.0)
    return output_dir


def run_scenario(classifier: CedClassifier, scenario_dir: Path, name: str) -> RunResult:
    from main import run_sources

    sources = [
        FileSource(room, scenario_dir / f"{name}_{room}.wav", end_behavior="stop")
        for room in SCENARIOS[name]
    ]
    engine = NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True))
    return run_sources(
        sources,
        classifier,
        engine,
        producer_options={
            "start_timestamp": DAY_START,
            "realtime_pacing": False,
            "stop_when_exhausted": True,
            "duration_sec": None,
        },
        notifier=None,
        logger=None,
    )


def test_every_scenario_has_an_expectation() -> None:
    assert set(EXPECTED_RULE_COUNTS) == set(SCENARIOS)


@pytest.mark.parametrize("name", list(EXPECTED_RULE_COUNTS))
def test_scenario_alert_rules_and_counts(
    classifier: CedClassifier, scenario_dir: Path, name: str
) -> None:
    result = run_scenario(classifier, scenario_dir, name)
    actual = dict(Counter(alert.rule for alert in result.alerts))
    assert actual == EXPECTED_RULE_COUNTS[name]


def test_s1_r2_reports_room_breakdown(
    classifier: CedClassifier, scenario_dir: Path
) -> None:
    result = run_scenario(classifier, scenario_dir, "S1_footsteps_x3")
    warning = next(alert for alert in result.alerts if alert.rule == "R2")
    assert warning.room_counts == {"거실": 3}
    assert "거실 3회" in warning.message


def test_s5_location_follows_louder_room(
    classifier: CedClassifier, scenario_dir: Path
) -> None:
    result = run_scenario(classifier, scenario_dir, "S5_two_rooms")
    mic_by_second = {
        round(frame.timestamp - DAY_START): frame.mic_name for frame in result.frames
    }
    # 거실 발소리(2~7초 구간), 안방 발소리(12~17초 구간) → 프레임 끝 시각 기준.
    assert all(mic_by_second[second] == "거실" for second in range(3, 9))
    assert all(mic_by_second[second] == "안방" for second in range(13, 19))
    room_alerts = [(alert.rule, alert.mic_name) for alert in result.alerts]
    assert ("R1", "거실") in room_alerts and ("R1", "안방") in room_alerts
