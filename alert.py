"""알림·상태 출력. 지금은 콘솔, Phase 2에서 OLED/LED로 교체한다.

decision.py가 같은 프레임에서 여러 Alert(예: R1과 R3)를 낼 수 있다. 사람이 보기에는 한 사건이므로
표시 단계에서만 1건으로 합친다. 원본 Alert 목록은 그대로 두고 CSV에 모두 기록한다.
"""

from dataclasses import dataclass
from datetime import datetime

from decision import CAUTION, LOCAL_TIMEZONE, WARNING
from models import Alert, Category, Frame

ALERT_BANNER_WIDTH = 72
SKIPPED_TEXT = "-"
LEVEL_RANK = {CAUTION: 0, WARNING: 1}
LEVEL_LABEL = {CAUTION: "주의", WARNING: "경고"}


@dataclass(frozen=True)
class MergedAlert:
    level: str  # 합쳐진 알림 중 최고 등급
    rules: tuple[str, ...]  # 발령 순서 그대로
    messages: tuple[str, ...]


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


class ConsoleNotifier:
    """매 초 상태 한 줄과, 눈에 띄는 알림 블록을 콘솔에 출력한다."""

    def show_status(self, frame: Frame, judged: Category | None) -> None:
        """대표 마이크, dB, top 라벨, 판정 카테고리를 한 줄로 출력한다 (side effect: 출력)."""
        judged_text = judged.value if judged else SKIPPED_TEXT
        print(
            f"{format_local_time(frame.timestamp)} | {frame.mic_name:4s} | "
            f"Leq {frame.leq_db:5.1f} Lmax {frame.lmax_db:5.1f} dB(A) | "
            f"{judged_text:8s} | {frame.top_label}"
        )

    def show_alerts(self, alerts: list[Alert]) -> None:
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
