import threading
import time

from pi_client.outputs import ConsoleOutput, OutputController
from protocol import StatusMessage


class RecordingOutput:
    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.cleared = threading.Event()

    def show_caution(self, alert: dict) -> None:
        self.events.append(("caution", time.monotonic()))

    def show_warning(self, alert: dict) -> None:
        self.events.append(("warning", time.monotonic()))

    def show_status(self, status: StatusMessage) -> None:
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


def test_console_output_shows_missing_mics() -> None:
    lines: list[str] = []
    output = ConsoleOutput(blink_count=2, blink_interval_sec=0.0, printer=lines.append)
    output.show_status(StatusMessage(("안방",), 4, 5, 0.0))
    output.show_caution(alert("caution"))
    assert lines[0] == "[마이크 이상] 안방 (활성 4/5)"
    assert lines.count("[LED] ● 켜짐") == 2
