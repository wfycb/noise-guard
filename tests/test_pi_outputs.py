import threading
import time

from pi_client.outputs import (
    CONNECTED,
    DISCONNECTED,
    ConsoleOutput,
    DisplayStatus,
    OutputController,
)
from protocol import StatusMessage


class RecordingOutput:
    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.cleared = threading.Event()

    def show_caution(self, alert: dict) -> None:
        self.events.append(("caution", time.monotonic()))

    def show_warning(self, alert: dict) -> None:
        self.events.append(("warning", time.monotonic()))

    def show_status(self, status: DisplayStatus) -> None:
        self.events.append(("status", time.monotonic()))

    def clear(self) -> None:
        self.events.append(("clear", time.monotonic()))
        self.cleared.set()


def alert(level: str) -> dict:
    return {
        "level": level,
        "rules": ["R1"],
        "room": "거실",
        "label_ko": "발소리",
        "count": 1,
        "peak_db": 60.0,
        "message": "m",
    }


def test_caution_and_warning_go_to_different_outputs() -> None:
    output = RecordingOutput()
    controller = OutputController(output, display_hold_sec=0.2)
    controller.start()
    controller.submit_alert(alert("caution"))
    controller.submit_alert(alert("warning"))
    assert output.cleared.wait(2.0)
    controller.stop()
    assert [kind for kind, _ in output.events] == ["caution", "warning", "clear"]


def test_warning_is_held_then_cleared() -> None:
    output = RecordingOutput()
    controller = OutputController(output, display_hold_sec=0.3)
    controller.start()
    controller.submit_alert(alert("warning"))
    assert output.cleared.wait(2.0)
    controller.stop()
    shown, cleared = output.events[0][1], output.events[1][1]
    assert cleared - shown >= 0.29


def test_new_warning_extends_hold() -> None:
    output = RecordingOutput()
    controller = OutputController(output, display_hold_sec=0.4)
    controller.start()
    controller.submit_alert(alert("warning"))
    time.sleep(0.25)
    controller.submit_alert(alert("warning"))
    assert output.cleared.wait(2.0)
    controller.stop()
    kinds = [kind for kind, _ in output.events]
    assert kinds == ["warning", "warning", "clear"]
    assert output.events[2][1] - output.events[1][1] >= 0.39


def test_output_failure_does_not_stop_controller() -> None:
    class BrokenOutput(RecordingOutput):
        def show_caution(self, alert: dict) -> None:
            raise RuntimeError("LED 고장")

    output = BrokenOutput()
    controller = OutputController(output, display_hold_sec=0.1)
    controller.start()
    controller.submit_alert(alert("caution"))
    controller.submit_status(StatusMessage(("안방",), 4, 5, 0.0))
    controller.stop()
    assert [kind for kind, _ in output.events] == ["status"]


def display(
    connection: str = CONNECTED,
    disconnected_sec: float = 0.0,
    restored: bool = False,
    missing: tuple[str, ...] | None = None,
    stale: bool = False,
) -> DisplayStatus:
    mic = None if missing is None else StatusMessage(missing, 5 - len(missing), 5, 0.0)
    return DisplayStatus(connection, disconnected_sec, restored, mic, stale)


def test_console_output_shows_missing_mics() -> None:
    lines: list[str] = []
    output = ConsoleOutput(blink_count=2, blink_interval_sec=0.0, printer=lines.append)
    output.show_status(display(missing=("안방",)))
    output.show_caution(alert("caution"))
    assert lines[0] == "[마이크 이상] 안방 (활성 4/5)"
    assert lines.count("[LED] ● 켜짐") == 2


def test_console_output_marks_old_mic_info_unknown_while_disconnected() -> None:
    lines: list[str] = []
    output = ConsoleOutput(printer=lines.append)
    output.show_status(
        display(DISCONNECTED, disconnected_sec=12.3, missing=("안방",), stale=True)
    )
    assert lines == [
        "[서버 연결 끊김] 12초째 — 소음 감지 중단",
        "[마이크 상태] 확인 불가 (마지막 수신: 안방 이상)",
    ]


def test_console_output_shows_restore() -> None:
    lines: list[str] = []
    ConsoleOutput(printer=lines.append).show_status(
        display(restored=True, missing=(), stale=True)
    )
    assert lines == ["[서버 연결 복구]", "[마이크 상태] 확인 불가"]


class StatusRecorder(RecordingOutput):
    def __init__(self) -> None:
        super().__init__()
        self.statuses: list[DisplayStatus] = []

    def show_status(self, status: DisplayStatus) -> None:
        self.statuses.append(status)


def test_controller_combines_connection_and_mic_status() -> None:
    output = StatusRecorder()
    controller = OutputController(output, display_hold_sec=1.0, refresh_sec=0.2)
    controller.start()
    controller.submit_connection(True)  # 처음 연결: 표시할 변화 없음
    controller.submit_status(StatusMessage(("안방",), 4, 5, 0.0))
    controller.submit_connection(False)
    time.sleep(0.5)  # 끊긴 동안 경과 시간을 다시 표시한다
    controller.submit_connection(False)  # 같은 상태는 무시
    controller.submit_connection(True)
    controller.submit_status(StatusMessage((), 5, 5, 1.0))
    controller.stop()
    summary = [
        (status.connection, status.restored, status.mic_status_stale)
        for status in output.statuses
    ]
    assert summary[0] == (CONNECTED, False, False)  # 마이크 이상 수신
    assert summary[1] == (DISCONNECTED, False, True)
    assert all(item == (DISCONNECTED, False, True) for item in summary[2:-2])
    assert len(summary) >= 5  # 끊긴 동안 refresh가 최소 한 번
    assert summary[-2] == (CONNECTED, True, True)  # 복구, 마이크 정보는 아직 확인 불가
    assert summary[-1] == (CONNECTED, False, False)  # 새 STATUS
    assert output.statuses[1].mic_status.missing_mics == ("안방",)
    assert output.statuses[-3].disconnected_sec >= 0.2
