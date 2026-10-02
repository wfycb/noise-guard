from mic_health import MicPresenceTracker

ROOMS = ["거실", "안방", "서재"]


def make_tracker(warn_ticks: int = 3) -> MicPresenceTracker:
    return MicPresenceTracker(ROOMS, warn_ticks=warn_ticks, hop_sec=1.0)


def test_no_change_while_all_present() -> None:
    tracker = make_tracker()
    for tick in range(5):
        assert tracker.update(float(tick), set(ROOMS)) is None
    assert tracker.last_active_count == 3


def test_warns_once_at_threshold() -> None:
    tracker = make_tracker()
    others = {"거실", "서재"}
    assert tracker.update(1.0, others) is None
    assert tracker.update(2.0, others) is None
    change = tracker.update(3.0, others)
    assert change is not None
    assert change.newly_missing == ("안방",)
    assert change.missing_mics == ("안방",)
    assert change.missing_seconds == {"안방": 3.0}
    assert (change.active_mics, change.total_mics) == (2, 3)
    assert tracker.update(4.0, others) is None  # 이미 경고한 상태는 다시 알리지 않는다


def test_recovery_reports_missing_duration() -> None:
    tracker = make_tracker()
    for tick in range(5):
        tracker.update(float(tick), {"거실", "서재"})
    change = tracker.update(5.0, set(ROOMS))
    assert change is not None
    assert change.recovered == ("안방",)
    assert change.missing_mics == ()
    assert change.missing_seconds == {"안방": 5.0}


def test_short_gap_below_threshold_is_silent() -> None:
    tracker = make_tracker()
    assert tracker.update(1.0, {"거실", "서재"}) is None
    assert tracker.update(2.0, {"거실", "서재"}) is None
    assert tracker.update(3.0, set(ROOMS)) is None
    assert tracker.missing_tick_counts()["안방"] == 2


def test_total_missing_ticks_accumulate_across_episodes() -> None:
    tracker = make_tracker()
    pattern = [{"거실"}] * 4 + [set(ROOMS)] + [{"거실", "안방"}] * 3
    for tick, present in enumerate(pattern):
        tracker.update(float(tick), present)
    assert tracker.missing_tick_counts() == {"거실": 0, "안방": 4, "서재": 7}
