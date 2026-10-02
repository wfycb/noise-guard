"""Pi 출력: HardwareOutput 인터페이스, 콘솔 구현, GPIO LED·디스플레이 뼈대, 출력 전용 스레드.

서버는 구조화된 ALERT/STATUS만 보내고 화면 배치는 여기서 정한다(디스플레이 모델 미정).
출력은 OutputController의 별도 스레드에서 하므로, 점멸·표시 유지 중에도 수집과 전송은 계속된다.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from pi_client import client_config
from protocol import StatusMessage

logger = logging.getLogger(__name__)

AlertPayload = dict[str, Any]


class HardwareOutput(Protocol):
    def show_caution(self, alert: AlertPayload) -> None:
        """caution: LED를 LED_BLINK_COUNT번 점멸."""

    def show_warning(self, alert: AlertPayload) -> None:
        """warning: 디스플레이에 방·소리 종류·횟수 표시 + LED 점등."""

    def show_status(self, status: StatusMessage) -> None:
        """마이크 상태(빠진 마이크). LED 패턴이나 디스플레이 구석 표시."""

    def clear(self) -> None:
        """warning 표시를 지우고 LED를 끈다."""


def describe_alert(alert: AlertPayload) -> str:
    """디스플레이 한 줄 요약: 방 · 소리 · 횟수 (규칙)."""
    return (
        f"{alert['room']} · {alert['label_ko']} · {alert['count']}회 "
        f"({'+'.join(alert['rules'])}, 최대 {alert['peak_db']} dB(A))"
    )


class ConsoleOutput:
    """LED 점멸과 디스플레이 내용을 콘솔로 흉내 낸다 (side effect: 출력)."""

    def __init__(
        self,
        blink_count: int = client_config.LED_BLINK_COUNT,
        blink_interval_sec: float = client_config.LED_BLINK_INTERVAL_SEC,
        printer: Callable[[str], None] = print,
    ) -> None:
        self._blink_count = blink_count
        self._blink_interval_sec = blink_interval_sec
        self._print = printer

    def show_caution(self, alert: AlertPayload) -> None:
        for _ in range(self._blink_count):
            self._print("[LED] ● 켜짐")
            time.sleep(self._blink_interval_sec)
            self._print("[LED] ○ 꺼짐")
            time.sleep(self._blink_interval_sec)
        self._print(f"[주의] {describe_alert(alert)}")

    def show_warning(self, alert: AlertPayload) -> None:
        self._print(
            "┌──────── 디스플레이 ────────\n"
            f"│ [경고] {describe_alert(alert)}\n"
            f"│ {alert['message']}\n"
            "└────────────────────────────\n"
            "[LED] ● 점등"
        )

    def show_status(self, status: StatusMessage) -> None:
        if status.missing_mics:
            self._print(
                f"[마이크 이상] {', '.join(status.missing_mics)} "
                f"(활성 {status.active_mics}/{status.total_mics})"
            )
        else:
            self._print(f"[마이크 정상] {status.active_mics}/{status.total_mics}")

    def clear(self) -> None:
        self._print("[디스플레이] 지움, [LED] 꺼짐")


class GpioLedOutput:
    """GPIO LED. TODO: 핀 번호가 정해지면 구현한다(gpiozero 등은 아직 의존성에 넣지 않음)."""

    def show_caution(self, alert: AlertPayload) -> None:
        raise NotImplementedError("GPIO LED 점멸 미구현 (핀 번호 미정)")

    def show_warning(self, alert: AlertPayload) -> None:
        raise NotImplementedError("GPIO LED 점등 미구현 (핀 번호 미정)")

    def show_status(self, status: StatusMessage) -> None:
        raise NotImplementedError("마이크 이상 LED 패턴 미구현")

    def clear(self) -> None:
        raise NotImplementedError("GPIO LED 끄기 미구현")


class DisplayOutput:
    """OLED 1.3" 또는 2.4" SPI TFT. TODO: 모델이 정해지면 구현한다.

    SSD1306 계열 OLED는 기본 폰트로 한글을 그리지 못하므로 한글 비트맵/TTF 폰트가 필요하다.
    """

    def show_caution(self, alert: AlertPayload) -> None:
        raise NotImplementedError("디스플레이 미구현 (모델 미정)")

    def show_warning(self, alert: AlertPayload) -> None:
        raise NotImplementedError("디스플레이 미구현 (모델 미정)")

    def show_status(self, status: StatusMessage) -> None:
        raise NotImplementedError("디스플레이 상태 표시 미구현 (모델 미정)")

    def clear(self) -> None:
        raise NotImplementedError("디스플레이 미구현 (모델 미정)")


class OutputController:
    """출력 전용 스레드. 수신 스레드는 submit만 하고 바로 돌아간다 (side effect: 스레드).

    warning은 DISPLAY_HOLD_SEC 동안 유지한 뒤 clear()한다. 유지 중 새 warning이 오면 연장한다.
    """

    def __init__(
        self,
        output: HardwareOutput,
        display_hold_sec: float = client_config.DISPLAY_HOLD_SEC,
    ) -> None:
        self._output = output
        self._display_hold_sec = display_hold_sec
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._clear_deadline: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"{client_config.THREAD_NAME_PREFIX}-output",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """남은 출력을 처리하고 스레드를 끝낸다."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(client_config.THREAD_JOIN_TIMEOUT_SEC)

    def submit_alert(self, alert: AlertPayload) -> None:
        self._queue.put(("alert", alert))

    def submit_status(self, status: StatusMessage) -> None:
        self._queue.put(("status", status))

    def _run(self) -> None:
        while True:
            timeout = None
            if self._clear_deadline is not None:
                timeout = max(0.0, self._clear_deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._clear_deadline = None
                self._call(self._output.clear)
                continue
            if item is None:
                if self._clear_deadline is not None:
                    self._call(self._output.clear)
                return
            kind, payload = item
            if kind == "status":
                self._call(self._output.show_status, payload)
            elif payload["level"] == "warning":
                self._call(self._output.show_warning, payload)
                self._clear_deadline = time.monotonic() + self._display_hold_sec
            else:
                self._call(self._output.show_caution, payload)

    @staticmethod
    def _call(method: Callable[..., None], *arguments: Any) -> None:
        # 출력 장치 오류가 수집·전송까지 멈추게 하면 안 되므로 로그만 남긴다.
        try:
            method(*arguments)
        except Exception:
            logger.exception("출력 실패")
