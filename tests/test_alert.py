import pytest

from alert import ConsoleSink, NetworkSink, build_alert_payload, merge_alerts
from mic_health import MicStatusChange
from models import Alert, Category, Frame
from protocol import (
    ALERT_REQUIRED_FIELDS,
    Hello,
    MessageType,
    MicInfo,
    decode_alert,
    decode_status,
)


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


# --- ALERT 본문과 NetworkSink ---


def make_frame(timestamp: float = 1000.0, mic_name: str = "거실") -> Frame:
    probs = {category: 0.0 for category in Category}
    return Frame(timestamp, mic_name, 60.0, 65.0, probs, "Walk, footsteps")


def make_rich_alert(rule: str, level: str, count: int, room_counts: dict) -> Alert:
    return Alert(
        level=level,
        rule=rule,
        category=Category.IMPACT,
        mic_name="거실",
        peak_db=61.2 if rule != "R3" else 45.0,
        count=count,
        timestamp=1000.0,
        message=f"{rule} message",
        room_counts=room_counts,
    )


def test_alert_payload_merges_and_keeps_structure() -> None:
    alerts = [
        make_rich_alert("R1", "caution", 3, {"거실": 2, "안방": 1}),
        make_rich_alert("R2", "warning", 3, {"거실": 2, "안방": 1}),
        make_rich_alert("R3", "warning", 1, {}),
    ]
    payload = build_alert_payload(
        alerts, make_frame(), source_mic_index=0, source_seq=9
    )
    assert payload["level"] == "warning"
    assert payload["rules"] == ["R1", "R2", "R3"]
    assert payload["message"] == "R2 message"  # 최고 등급 중 첫 알림
    assert payload["label"] == "Walk, footsteps"
    assert payload["label_ko"] == "발소리"
    assert payload["peak_db"] == 61.2
    assert payload["count"] == 3
    assert payload["room_counts"] == {"거실": 2, "안방": 1}
    assert (payload["source_mic_index"], payload["source_seq"]) == (0, 9)
    assert set(ALERT_REQUIRED_FIELDS) <= set(payload)


def test_unknown_label_stays_english() -> None:
    frame = Frame(1.0, "거실", 60.0, 65.0, {}, "Sizzle")
    payload = build_alert_payload([make_alert("R1", "caution")], frame, 0, 0)
    assert payload["label_ko"] == "Sizzle"


class FakeSession:
    def __init__(self, sample_rate: int, chunk_samples: int) -> None:
        self.hello = Hello(1, "pi", (MicInfo(0, "거실", sample_rate),), chunk_samples)
        self.stream_start_time = 1000.0
        self.sent: list[tuple[MessageType, bytes]] = []

    def send(self, message_type: MessageType, body: bytes = b"") -> bool:
        self.sent.append((message_type, body))
        return True


def test_source_seq_points_to_chunk_holding_last_sample_of_frame() -> None:
    session = FakeSession(sample_rate=48000, chunk_samples=4800)
    sink = NetworkSink(session, hop_sec=1.0)  # type: ignore[arg-type]
    # tick 3이 끝나는 샘플은 144000 → 마지막 샘플 143999는 seq 29 (4800 × 30 = 144000).
    assert sink.source_position(make_frame(timestamp=1003.0)) == (0, 29)
    session_44k = FakeSession(sample_rate=44100, chunk_samples=4800)
    sink_44k = NetworkSink(session_44k, hop_sec=1.0)  # type: ignore[arg-type]
    # 44100 × 2 = 88200 → 마지막 샘플 88199 // 4800 = 18
    assert sink_44k.source_position(make_frame(timestamp=1002.0)) == (0, 18)


def test_network_sink_sends_one_alert_per_frame_and_skips_empty() -> None:
    session = FakeSession(48000, 4800)
    sink = NetworkSink(session, hop_sec=1.0)  # type: ignore[arg-type]
    sink.emit([], make_frame())
    sink.emit(
        [make_alert("R1", "caution"), make_alert("R3", "warning")],
        make_frame(1002.0),
    )
    assert [message_type for message_type, _ in session.sent] == [MessageType.ALERT]
    assert decode_alert(session.sent[0][1])["rules"] == ["R1", "R3"]


def test_mic_status_goes_to_console_and_network(capsys: pytest.CaptureFixture) -> None:
    change = MicStatusChange(
        timestamp=1003.0,
        missing_mics=("안방",),
        newly_missing=("안방",),
        recovered=(),
        missing_seconds={"안방": 3.0},
        active_mics=4,
        total_mics=5,
    )
    ConsoleSink().emit_mic_status(change)
    assert "[마이크 이상] 안방: 3초째 데이터 없음" in capsys.readouterr().out
    session = FakeSession(48000, 4800)
    NetworkSink(session, hop_sec=1.0).emit_mic_status(change)  # type: ignore[arg-type]
    message_type, body = session.sent[0]
    assert message_type == MessageType.STATUS
    assert decode_status(body).missing_mics == ("안방",)


def test_status_line_shows_active_mic_count(capsys: pytest.CaptureFixture) -> None:
    ConsoleSink().emit_frame(make_frame(), Category.IMPACT, 4, 5)
    assert "마이크 4/5" in capsys.readouterr().out
