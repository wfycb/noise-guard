from datetime import datetime
from itertools import pairwise

import pytest

from decision import (
    LOCAL_TIMEZONE,
    DecisionConfig,
    NoiseDecisionEngine,
    judge_category,
    windowed_leq_db,
)
from models import Alert, Category, Frame

THRESHOLDS = {Category.IMPACT: 0.2, Category.AIRBORNE: 0.3}
# 이벤트 테스트에서 R3(1분 Leq)가 끼어들지 않도록 프레임 Leq는 낮게 둔다.
QUIET_LEQ_DB = 20.0


def local_timestamp(hour: int, minute: int = 0, second: int = 0) -> float:
    return datetime(
        2026, 10, 2, hour, minute, second, tzinfo=LOCAL_TIMEZONE
    ).timestamp()


DAY_START = local_timestamp(14)
NIGHT_START = local_timestamp(23)


def make_probs(
    impact: float = 0.0, airborne: float = 0.0, excluded: float = 0.0
) -> dict[Category, float]:
    return {
        Category.IMPACT: impact,
        Category.AIRBORNE: airborne,
        Category.EXCLUDED: excluded,
        Category.OTHER: 0.0,
    }


def make_frame(
    timestamp: float,
    lmax_db: float = 30.0,
    leq_db: float = QUIET_LEQ_DB,
    impact: float = 0.0,
    airborne: float = 0.0,
    excluded: float = 0.0,
    mic_name: str = "거실",
) -> Frame:
    return Frame(
        timestamp=timestamp,
        mic_name=mic_name,
        leq_db=leq_db,
        lmax_db=lmax_db,
        category_probs=make_probs(impact, airborne, excluded),
        top_label="Walk, footsteps",
    )


def make_engine(demo_mode: bool = False) -> NoiseDecisionEngine:
    return NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=demo_mode))


def feed(engine: NoiseDecisionEngine, frames: list[Frame]) -> list[Alert]:
    alerts: list[Alert] = []
    for frame in frames:
        alerts.extend(engine.update(frame))
    return alerts


def rules(alerts: list[Alert]) -> list[str]:
    return [alert.rule for alert in alerts]


# --- judge_category ---


def test_both_below_threshold_is_skipped() -> None:
    assert judge_category(make_probs(impact=0.19, airborne=0.29), THRESHOLDS) is None


def test_thresholds_are_applied_per_category() -> None:
    # 0.25는 IMPACT 임계값(0.2)은 넘지만 AIRBORNE 임계값(0.3)은 못 넘는다.
    assert judge_category(make_probs(impact=0.25), THRESHOLDS) == Category.IMPACT
    assert judge_category(make_probs(airborne=0.25), THRESHOLDS) is None


def test_threshold_is_inclusive() -> None:
    assert judge_category(make_probs(impact=0.2), THRESHOLDS) == Category.IMPACT
    assert judge_category(make_probs(airborne=0.3), THRESHOLDS) == Category.AIRBORNE


def test_higher_probability_wins_when_both_pass() -> None:
    assert (
        judge_category(make_probs(impact=0.6, airborne=0.4), THRESHOLDS)
        == Category.IMPACT
    )
    assert (
        judge_category(make_probs(impact=0.25, airborne=0.5), THRESHOLDS)
        == Category.AIRBORNE
    )


def test_excluded_alone_is_skipped() -> None:
    assert judge_category(make_probs(excluded=0.95), THRESHOLDS) is None


def test_impact_wins_over_high_excluded() -> None:
    probs = make_probs(impact=0.3, excluded=0.9)
    assert judge_category(probs, THRESHOLDS) == Category.IMPACT


@pytest.mark.parametrize("missing", [Category.IMPACT, Category.AIRBORNE])
def test_missing_category_counts_as_zero(missing: Category) -> None:
    probs = make_probs(impact=0.5, airborne=0.5)
    del probs[missing]
    remaining = Category.AIRBORNE if missing == Category.IMPACT else Category.IMPACT
    assert judge_category(probs, THRESHOLDS) == remaining


# --- windowed Leq (9.5 (a)) ---


def test_windowed_leq_full_window_equals_level() -> None:
    assert windowed_leq_db([40.0] * 60, 60.0, 1.0, 0.0) == pytest.approx(40.0)


def test_windowed_leq_divides_by_whole_window() -> None:
    # 60초 창에 60 dB 프레임 1개 → 60 - 10*log10(60) ≈ 42.2 (바닥 0 dB 기여는 무시할 수준).
    assert windowed_leq_db([60.0], 60.0, 1.0, 0.0) == pytest.approx(42.2, abs=0.05)


# --- 엔진: 사양 11번 ---


def test_three_lmax_events_give_three_cautions_and_one_r2_warning() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + offset, lmax_db=60.0, impact=0.9)
        for offset in (0.0, 5.0, 10.0)
    ]
    alerts = feed(engine, frames)
    assert rules(alerts) == ["R1", "R1", "R1", "R2"]
    assert [alert.count for alert in alerts if alert.rule == "R1"] == [1, 2, 3]
    assert [alert.level for alert in alerts] == ["caution"] * 3 + ["warning"]


def test_consecutive_frames_merge_into_one_event() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=60.0, impact=0.9) for second in range(10)
    ]
    assert rules(feed(engine, frames)) == ["R1"]
    assert engine.stored_lmax_exceed_count == 1


def test_caution_fires_on_first_frame_exceeding_within_event() -> None:
    engine = make_engine()
    lmax_values = [50.0, 55.0, 58.0, 61.0]
    frames = [
        make_frame(DAY_START + index, lmax_db=lmax, impact=0.9)
        for index, lmax in enumerate(lmax_values)
    ]
    alerts = feed(engine, frames)
    assert rules(alerts) == ["R1"]
    assert alerts[0].timestamp == DAY_START + 2
    assert alerts[0].peak_db == 58.0


def test_water_alone_gives_no_alert_and_no_count() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=70.0, leq_db=70.0, excluded=0.95)
        for second in range(30)
    ]
    assert feed(engine, frames) == []
    assert engine.stored_lmax_exceed_count == 0
    assert engine.stored_frame_count == 0


def test_water_with_footsteps_counts_as_impact() -> None:
    engine = make_engine()
    frame = make_frame(DAY_START, lmax_db=60.0, impact=0.4, excluded=0.9)
    assert rules(engine.update(frame)) == ["R1"]
    assert engine.stored_lmax_exceed_count == 1


def test_same_lmax_below_day_limit_but_above_night_limit() -> None:
    day_engine = make_engine()
    night_engine = make_engine()
    assert day_engine.update(make_frame(DAY_START, lmax_db=55.0, impact=0.9)) == []
    night_alerts = night_engine.update(
        make_frame(NIGHT_START, lmax_db=55.0, impact=0.9)
    )
    assert rules(night_alerts) == ["R1"]


def test_limit_changes_at_22_boundary() -> None:
    engine = make_engine()
    before = make_frame(local_timestamp(21, 59, 55), lmax_db=55.0, impact=0.9)
    after = make_frame(local_timestamp(22, 0, 5), lmax_db=55.0, impact=0.9)
    assert engine.update(before) == []
    assert rules(engine.update(after)) == ["R1"]


def test_r2_cooldown_blocks_then_allows_rewarning() -> None:
    engine = make_engine()
    cooldown = DecisionConfig.from_config(demo_mode=False).cooldown_sec
    first_three = [
        make_frame(DAY_START + offset, lmax_db=60.0, impact=0.9)
        for offset in (0.0, 5.0, 10.0)
    ]
    assert rules(feed(engine, first_three)).count("R2") == 1
    within_cooldown = make_frame(DAY_START + 15.0, lmax_db=60.0, impact=0.9)
    assert rules(engine.update(within_cooldown)) == ["R1"]
    after_cooldown = make_frame(DAY_START + 10.0 + cooldown, lmax_db=60.0, impact=0.9)
    assert rules(engine.update(after_cooldown)) == ["R1", "R2"]


def test_r3_impact_leq_over_one_minute() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=40.0, leq_db=40.0, impact=0.9)
        for second in range(60)
    ]
    alerts = feed(engine, frames)
    assert rules(alerts) == ["R3"]
    assert alerts[0].level == "warning"
    assert alerts[0].peak_db >= 39.0


def test_r4_airborne_leq_over_five_minutes() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=46.0, leq_db=46.0, airborne=0.8)
        for second in range(300)
    ]
    alerts = feed(engine, frames)
    assert alerts and set(rules(alerts)) == {"R4"}
    assert alerts[0].category == Category.AIRBORNE
    # 소음이 이어지면 쿨다운마다 재경고한다.
    cooldown = DecisionConfig.from_config(demo_mode=False).cooldown_sec
    gaps = [later.timestamp - earlier.timestamp for earlier, later in pairwise(alerts)]
    assert all(gap >= cooldown for gap in gaps)


def test_r4_not_triggered_below_limit() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, leq_db=44.0, airborne=0.8)
        for second in range(300)
    ]
    assert feed(engine, frames) == []


def test_old_data_is_pruned() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=60.0, leq_db=30.0, impact=0.9)
        for second in range(0, 600, 10)
    ]
    feed(engine, frames)
    assert engine.stored_frame_count == 60
    longest_window = DecisionConfig.from_config(demo_mode=False).lmax_window_sec
    engine.update(make_frame(DAY_START + 600 + longest_window, airborne=0.0))
    assert engine.stored_frame_count == 0
    assert engine.stored_lmax_exceed_count == 0


@pytest.mark.parametrize(
    "frame",
    [
        Frame(DAY_START, "거실", 40.0, 45.0, {}, ""),
        make_frame(DAY_START, lmax_db=-20.0, leq_db=-30.0, impact=0.9),
        make_frame(DAY_START, lmax_db=150.0, leq_db=150.0, impact=0.9),
        make_frame(DAY_START, lmax_db=150.0, leq_db=150.0, airborne=0.9),
        make_frame(DAY_START, lmax_db=float("nan"), leq_db=float("nan"), impact=0.9),
    ],
)
def test_unusual_frames_do_not_raise(frame: Frame) -> None:
    make_engine().update(frame)


def test_demo_mode_shortens_windows() -> None:
    demo = DecisionConfig.from_config(demo_mode=True)
    normal = DecisionConfig.from_config(demo_mode=False)
    assert demo.lmax_window_sec < normal.lmax_window_sec
    assert demo.impact_leq_window_sec < normal.impact_leq_window_sec
    assert demo.airborne_leq_window_sec < normal.airborne_leq_window_sec
    assert demo.lmax_count_to_warn == normal.lmax_count_to_warn


def test_r2_counts_whole_house_and_reports_rooms() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START, lmax_db=60.0, impact=0.9, mic_name="거실"),
        make_frame(DAY_START + 5.0, lmax_db=60.0, impact=0.9, mic_name="안방"),
        make_frame(DAY_START + 10.0, lmax_db=60.0, impact=0.9, mic_name="거실"),
    ]
    alerts = feed(engine, frames)
    warning = next(alert for alert in alerts if alert.rule == "R2")
    assert warning.count == 3
    assert warning.room_counts == {"거실": 2, "안방": 1}
    assert "거실 2회, 안방 1회" in warning.message
    assert all(alert.room_counts for alert in alerts)


def test_leq_alerts_have_no_room_counts() -> None:
    engine = make_engine()
    frames = [
        make_frame(DAY_START + second, lmax_db=40.0, leq_db=40.0, impact=0.9)
        for second in range(60)
    ]
    alerts = feed(engine, frames)
    assert rules(alerts) == ["R3"]
    assert alerts[0].room_counts == {}
