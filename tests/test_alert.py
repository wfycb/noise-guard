from alert import merge_alerts
from models import Alert, Category


def make_alert(rule: str, level: str) -> Alert:
    return Alert(
        level=level,
        rule=rule,
        category=Category.IMPACT,
        mic_name="거실",
        peak_db=60.0,
        count=1,
        timestamp=0.0,
        message=f"{rule} message",
    )


def test_no_alerts_merge_to_none() -> None:
    assert merge_alerts([]) is None


def test_merge_takes_highest_level_and_keeps_rule_order() -> None:
    merged = merge_alerts([make_alert("R1", "caution"), make_alert("R3", "warning")])
    assert merged is not None
    assert merged.level == "warning"
    assert merged.rules == ("R1", "R3")
    assert merged.messages == ("R1 message", "R3 message")


def test_caution_only_stays_caution() -> None:
    merged = merge_alerts([make_alert("R1", "caution")])
    assert merged is not None
    assert merged.level == "caution"
    assert merged.rules == ("R1",)
