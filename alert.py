"""알림·상태 출력: AlertSink 인터페이스와 콘솔·네트워크 구현.

decision.py가 같은 프레임에서 여러 Alert(예: R1과 R3)를 낼 수 있다. 사람이 보기에는 한 사건이므로
표시·송신 단계에서만 1건으로 합친다. 원본 Alert 목록은 그대로 두고 CSV에 모두 기록한다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from decision import CAUTION, LOCAL_TIMEZONE, WARNING
from label_ko import to_korean
from mic_health import MicStatusChange
from models import Alert, Category, Frame
from protocol import MessageType, StatusMessage, encode_alert, encode_status
from server_net import ClientSession

ALERT_BANNER_WIDTH = 72
SKIPPED_TEXT = "-"
LEVEL_RANK = {CAUTION: 0, WARNING: 1}
LEVEL_LABEL = {CAUTION: "주의", WARNING: "경고"}


@dataclass(frozen=True)
class MergedAlert:
    level: str  # 합쳐진 알림 중 최고 등급
    rules: tuple[str, ...]  # 발령 순서 그대로
    messages: tuple[str, ...]


class AlertSink(Protocol):
    """판단 결과를 내보내는 곳. 콘솔, 네트워크(Pi), 나중에는 다른 출력이 될 수 있다."""

    def emit(self, alerts: list[Alert], frame: Frame) -> None:
        """한 프레임에서 나온 알림들. 비어 있으면 아무것도 하지 않는다."""

    def emit_frame(
        self,
        frame: Frame,
        judged: Category | None,
        active_mics: int,
        total_mics: int,
    ) -> None:
        """매 프레임 상태."""

    def emit_mic_status(self, change: MicStatusChange) -> None:
        """마이크 빠짐/복구 (상태가 바뀔 때만 불린다)."""


def format_local_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, LOCAL_TIMEZONE).strftime("%H:%M:%S")


def merge_alerts(alerts: list[Alert]) -> MergedAlert | None:
    """같은 프레임의 알림들을 1건으로 합친다. 등급은 warning > caution."""
    if not alerts:
        return None
    highest_level = max(
        (alert.level for alert in alerts), key=lambda level: LEVEL_RANK[level]
    )
    return MergedAlert(
        level=highest_level,
        rules=tuple(alert.rule for alert in alerts),
        messages=tuple(alert.message for alert in alerts),
    )


def build_alert_payload(
    alerts: list[Alert], frame: Frame, source_mic_index: int, source_seq: int
) -> dict[str, Any]:
    """합친 알림을 ALERT JSON 본문으로 만든다 (순수 함수). 화면 배치는 클라이언트가 정한다."""
    merged = merge_alerts(alerts)
    if merged is None:
        raise ValueError("알림이 없으면 ALERT를 만들 수 없습니다")
    # 대표 알림: 가장 높은 등급 중 처음 것. 메시지와 카테고리를 여기서 가져온다.
    lead = next(alert for alert in alerts if alert.level == merged.level)
    room_counts = next((alert.room_counts for alert in alerts if alert.room_counts), {})
    return {
        "level": merged.level,
        "rules": list(merged.rules),
        "room": frame.mic_name,
        "category": lead.category.value,
        "label": frame.top_label,
        "label_ko": to_korean(frame.top_label),
        "peak_db": round(max(alert.peak_db for alert in alerts), 1),
        "count": max(alert.count for alert in alerts),
        "room_counts": dict(room_counts),
        "message": lead.message,
        "timestamp": frame.timestamp,
        "source_mic_index": source_mic_index,
        "source_seq": source_seq,
    }


class ConsoleSink:
    """매 초 상태 한 줄과, 눈에 띄는 알림·마이크 이상 블록을 콘솔에 출력한다."""

    def emit_frame(
        self,
        frame: Frame,
        judged: Category | None,
        active_mics: int,
        total_mics: int,
    ) -> None:
        """대표 마이크, 활성 마이크 수, dB, 판정, top 라벨 한 줄 (side effect: 출력)."""
        judged_text = judged.value if judged else SKIPPED_TEXT
        print(
            f"{format_local_time(frame.timestamp)} | 마이크 {active_mics}/{total_mics} | "
            f"{frame.mic_name:4s} | "
            f"Leq {frame.leq_db:5.1f} Lmax {frame.lmax_db:5.1f} dB(A) | "
            f"{judged_text:8s} | {frame.top_label}"
        )

    def emit(self, alerts: list[Alert], frame: Frame) -> None:
        """한 프레임의 알림들을 1건으로 합쳐 출력한다 (side effect: 출력).

        caution만 있으면 한 줄 강조, warning이 하나라도 있으면 테두리 블록.
        """
        merged = merge_alerts(alerts)
        if merged is None:
            return
        header = f"[{LEVEL_LABEL[merged.level]}] {'+'.join(merged.rules)}"
        details = "\n".join(f"   - {message}" for message in merged.messages)
        if merged.level == WARNING:
            border = "!" * ALERT_BANNER_WIDTH
            print(f"{border}\n!! {header}\n{details}\n{border}")
        else:
            print(f">> {header}\n{details}")

    def emit_mic_status(self, change: MicStatusChange) -> None:
        """마이크 빠짐은 테두리 블록, 복구는 한 줄로 출력한다 (side effect: 출력)."""
        for room in change.newly_missing:
            border = "#" * ALERT_BANNER_WIDTH
            print(
                f"{border}\n## [마이크 이상] {room}: "
                f"{change.missing_seconds[room]:g}초째 데이터 없음 "
                f"(활성 {change.active_mics}/{change.total_mics})\n{border}"
            )
        for room in change.recovered:
            print(
                f"## [마이크 복구] {room}: 데이터 다시 수신 "
                f"({change.missing_seconds[room]:g}초 동안 빠짐)"
            )


class NetworkSink:
    """합친 알림을 ALERT로, 마이크 상태 변화를 STATUS로 Pi에 보낸다 (side effect: 송신 큐).

    연결이 이미 끊겼으면 버리고 센다. 송신은 세션의 송신 스레드가 하므로 판단 루프가 막히지 않는다.
    """

    def __init__(self, session: ClientSession, hop_sec: float) -> None:
        self._session = session
        self._hop_sec = hop_sec
        self._mics = {mic.room: mic for mic in session.hello.mics}
        self.sent_alerts = 0
        self.sent_statuses = 0
        self.dropped_messages = 0

    def source_position(self, frame: Frame) -> tuple[int, int]:
        """프레임 마지막 샘플이 들어 있는 대표 마이크 청크의 (mic_index, seq)."""
        mic = self._mics[frame.mic_name]
        tick = round(
            (frame.timestamp - self._session.stream_start_time) / self._hop_sec
        )
        end_sample = tick * round(self._hop_sec * mic.sample_rate)
        seq = max(end_sample - 1, 0) // self._session.hello.chunk_samples
        return mic.index, seq

    def emit(self, alerts: list[Alert], frame: Frame) -> None:
        if not alerts:
            return
        mic_index, seq = self.source_position(frame)
        body = encode_alert(build_alert_payload(alerts, frame, mic_index, seq))
        if self._session.send(MessageType.ALERT, body):
            self.sent_alerts += 1
        else:
            self.dropped_messages += 1

    def emit_frame(
        self,
        frame: Frame,
        judged: Category | None,
        active_mics: int,
        total_mics: int,
    ) -> None:
        """매 프레임 상태는 보내지 않는다(대역폭·Pi 화면 갱신 부담). 변화만 STATUS로 보낸다."""

    def emit_mic_status(self, change: MicStatusChange) -> None:
        status = StatusMessage(
            change.missing_mics, change.active_mics, change.total_mics, change.timestamp
        )
        if self._session.send(MessageType.STATUS, encode_status(status)):
            self.sent_statuses += 1
        else:
            self.dropped_messages += 1
