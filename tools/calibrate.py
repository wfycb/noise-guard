"""소음계 보정 도구 (side effect: 오디오 장치·소켓 접근, 콘솔 입출력, calibration.json 쓰기).

마이크별로 dBFS(A) → dB(A) 오프셋을 실측해 calibration.json에 저장한다. 레벨 계산은 서버와 같은
level.LevelMeter(마이크 원래 샘플레이트에서 A특성, 125ms 블록, 블록 사이 필터 상태 유지)를 쓴다.

절차 (방 하나, 마이크 하나씩):
    1. 방을 조용하게 두고 배경 소음을 CALIBRATION_MEASURE_SEC 동안 잰다.
    2. 스피커로 핑크노이즈를 일정하게 재생한다.
    3. 소음계를 마이크 바로 옆에 두고 A특성, Leq 모드로 맞춘다.
       Leq 모드가 없으면 Fast로 여러 번 읽어 평균한다.
    4. CALIBRATION_MEASURE_SEC 동안 프로그램 Leq를 잰다.
    5. 같은 시간의 소음계 값을 입력한다.
    6. offset = 소음계 Leq − 프로그램 Leq.
    프로그램 Leq − 배경 Leq가 CALIBRATION_MIN_SNR_DB(10 dB)보다 작으면 경고하고 그 단계를 다시 잴지 묻는다.
    --levels N이면 볼륨을 바꿔 4~6을 N번 하고, 오프셋 차이가 CALIBRATION_MAX_SPREAD_DB를 넘으면 경고한다.

실행 (프로젝트 루트에서):
    # 노트북에 꽂은 마이크
    .venv\\Scripts\\python -m tools.calibrate --mic 거실 --source mic --device 18 --levels 3
    # Pi 마이크: 이 도구가 서버가 되고 Pi에서 pi_client를 실행한다
    .venv\\Scripts\\python -m tools.calibrate --mic 거실 --source network --port 5000
"""

import argparse
import logging
import math
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import config
from calibration import (
    LevelReading,
    MicCalibration,
    build_calibration,
    evaluate_background,
    make_reading,
    save_mic_calibration,
    signal_to_background_warning,
    spread_warning,
)
from capture import AudioSource
from classifier import judgment_limits_db
from decision import LOCAL_TIMEZONE
from level import LevelMeter, leq_db

MONITOR_POLL_SEC = 0.05


class LevelMonitor(threading.Thread):
    """소스를 계속 진행시키며 125ms 블록 레벨(dBFS(A))을 모은다 (side effect: 스레드, 장치 대기).

    측정 사이(사람이 소음계 값을 입력하는 동안)에도 계속 돌려야 네트워크 소스의 수신 버퍼가
    밀리지 않는다. 측정 길이는 벽시계가 아니라 받은 오디오 블록 수로 센다.
    """

    def __init__(self, sources: list[AudioSource], realtime_pacing: bool) -> None:
        super().__init__(name=f"{config.THREAD_NAME_PREFIX}-calibrate", daemon=True)
        self._sources = sources
        self._realtime_pacing = realtime_pacing
        self._meters = {
            source.name: LevelMeter(source.sample_rate) for source in sources
        }
        self._total_read = {source.name: 0 for source in sources}
        self._collected: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

    def run(self) -> None:
        hop_sec = config.CLASSIFY_HOP_SEC
        while not self.stop_event.is_set():
            if self._realtime_pacing and self.stop_event.wait(hop_sec):
                return
            for source in self._sources:
                source.advance(hop_sec)
                samples, self._total_read[source.name] = source.read_new(
                    self._total_read[source.name]
                )
                levels = self._meters[source.name].push(samples)
                with self._lock:
                    if source.name in self._collected:
                        self._collected[source.name].extend(levels)
            if all(source.exhausted for source in self._sources):
                return

    def measure(self, room: str, duration_sec: float) -> float:
        """room의 다음 duration_sec 분량 오디오의 Leq(dBFS(A))를 잰다 (side effect: 대기)."""
        needed_blocks = math.ceil(duration_sec / config.FAST_BLOCK_SEC)
        with self._lock:
            self._collected[room] = []
        while True:
            with self._lock:
                levels = self._collected[room]
                if len(levels) >= needed_blocks:
                    del self._collected[room]
                    return leq_db(levels[:needed_blocks])
            if not self.is_alive():
                raise RuntimeError(f"{room}: 측정 중 오디오 입력이 끝났습니다")
            time.sleep(MONITOR_POLL_SEC)


def parse_meter_value(text: str) -> float:
    """소음계 값 입력 → dB(A). 숫자가 아니거나 범위를 벗어나면 ValueError (순수 함수)."""
    value = float(text.strip().replace("dB", "").replace("(A)", ""))
    if not 0.0 < value < 140.0:
        raise ValueError(f"소음계 값 {value}가 범위(0~140 dB(A))를 벗어났습니다")
    return value


YES_ANSWERS = ("", "y", "yes", "예", "네", "ㅇ")
NO_ANSWERS = ("n", "no", "아니오", "아니요", "ㄴ")


def parse_yes_no(text: str) -> bool:
    """Y/n 답. 빈 입력은 예(기본값). 알 수 없는 답은 ValueError (순수 함수)."""
    answer = text.strip().lower()
    if answer in YES_ANSWERS:
        return True
    if answer in NO_ANSWERS:
        return False
    raise ValueError(f"'{text}'는 예/아니오로 알아들을 수 없습니다")


def ask_yes_no(prompt: str, read_line: Callable[[str], str] = input) -> bool:
    """예/아니오를 알아들을 때까지 다시 묻는다 (side effect: 콘솔 입력)."""
    while True:
        try:
            return parse_yes_no(read_line(prompt))
        except ValueError as error:
            print(f"  다시 입력하세요: {error}")


def ask_meter_value(prompt: str, read_line: Callable[[str], str] = input) -> float:
    """소음계 값을 숫자가 들어올 때까지 다시 묻는다 (side effect: 콘솔 입력)."""
    while True:
        try:
            return parse_meter_value(read_line(prompt))
        except ValueError as error:
            print(f"  다시 입력하세요: {error}")


def run_calibration(
    monitor: LevelMonitor,
    room: str,
    level_count: int,
    duration_sec: float,
    sample_rate: int,
    device: str,
    read_line: Callable[[str], str] = input,
) -> MicCalibration:
    """배경 → 볼륨별 핑크노이즈 측정 순서로 진행한다 (side effect: 콘솔 입출력, 대기)."""
    read_line(
        f"[1] {room}: 방을 조용하게 두고 Enter를 누르면 배경 소음을 {duration_sec:g}초 잽니다 "
    )
    background_dbfs_a = monitor.measure(room, duration_sec)
    print(f"    배경 소음: {background_dbfs_a:.1f} dBFS(A)")
    readings: list[LevelReading] = []
    for level_index in range(1, level_count + 1):
        read_line(
            f"[{level_index + 1}] 핑크노이즈를 재생하세요 (볼륨 {level_index}/{level_count}). "
            f"소음계는 A특성 Leq 모드로 맞추고, Enter를 누르는 순간부터 {duration_sec:g}초 동안 "
            "소음계도 함께 측정하세요 "
        )
        program_leq = monitor.measure(room, duration_sec)
        while (
            warning := signal_to_background_warning(
                program_leq, background_dbfs_a, config.CALIBRATION_MIN_SNR_DB
            )
        ) is not None:
            print(f"[경고] {warning}")
            if not ask_yes_no("    이 단계를 다시 측정할까요? [Y/n] ", read_line):
                break
            read_line(
                f"    볼륨을 올리고 Enter를 누르면 {duration_sec:g}초 동안 다시 잽니다 "
                "(소음계도 새로 측정) "
            )
            program_leq = monitor.measure(room, duration_sec)
        meter_leq = ask_meter_value(
            f"    프로그램 {program_leq:.1f} dBFS(A). 같은 시간의 소음계 Leq(dB(A)): ",
            read_line,
        )
        reading = make_reading(meter_leq, program_leq)
        readings.append(reading)
        print(f"    오프셋 {reading.offset_db:.1f} dB")
    measured_at = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds")
    return build_calibration(
        readings, background_dbfs_a, sample_rate, device, measured_at
    )


def report_calibration(room: str, calibration: MicCalibration) -> list[str]:
    """결과와 검증 경고를 출력하고 경고 목록을 반환한다 (side effect: 콘솔 출력)."""
    warnings = []
    spread = spread_warning(list(calibration.levels), config.CALIBRATION_MAX_SPREAD_DB)
    if spread:
        warnings.append(spread)
    background_warnings, recommended_gate = evaluate_background(
        calibration.background_leq_db,
        config.SKIP_CLASSIFY_BELOW_DB,
        min(judgment_limits_db()),
        config.SKIP_MARGIN_DB,
        config.CALIBRATION_BACKGROUND_NEAR_LIMIT_DB,
        config.CALIBRATION_GATE_HEADROOM_DB,
    )
    warnings.extend(background_warnings)
    print(f"\n===== {room} 보정 결과 =====")
    print(
        f"오프셋: {calibration.offset_db:.1f} dB (볼륨 {len(calibration.levels)}단계 평균)"
    )
    for reading in calibration.levels:
        print(
            f"  소음계 {reading.meter_leq:.1f} dB(A) − 프로그램 {reading.program_leq:.1f} "
            f"dBFS(A) = {reading.offset_db:.1f} dB"
        )
    print(f"배경 소음: {calibration.background_leq_db:.1f} dB(A)")
    print(
        f"권장 게이트: {recommended_gate:.1f} dB(A) "
        f"(현재 SKIP_CLASSIFY_BELOW_DB = {config.SKIP_CLASSIFY_BELOW_DB}, 자동으로 바꾸지 않음)"
    )
    for warning in warnings:
        print(f"[경고] {warning}")
    return warnings


def open_local_source(room: str, device: int | str) -> tuple[list[AudioSource], bool]:
    """로컬 마이크 (side effect: 장치 열기)."""
    from capture import MicSource

    source = MicSource(room, device)
    source.start()
    return [source], True


def open_network_sources(
    room: str, host: str, port: int
) -> tuple[list[AudioSource], bool, object]:
    """Pi 연결을 기다려 그 마이크들을 쓴다 (side effect: 소켓)."""
    from server_net import AudioServer

    server = AudioServer(host, port)
    server.start()
    print(f"Pi 연결 대기 중 {server.address[0]}:{server.address[1]} ...")
    session = None
    while session is None:
        session = server.next_session(timeout=config.SERVER_THREAD_POLL_SEC)
    rooms = [source.name for source in session.sources]
    if room not in rooms:
        server.close()
        raise SystemExit(f"Pi 마이크에 '{room}'이 없습니다: {rooms}")
    return list(session.sources), False, server


def parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="소음계 보정")
    parser.add_argument("--mic", required=True, help="보정할 방 이름")
    parser.add_argument("--source", choices=["mic", "network"], default="mic")
    parser.add_argument(
        "--device", type=parse_device, help="로컬 장치 (기본: MIC_DEVICES)"
    )
    parser.add_argument(
        "--host", default=config.SERVER_BIND_HOST, help="network 모드 bind 주소"
    )
    parser.add_argument("--port", type=int, default=config.SERVER_PORT)
    parser.add_argument("--levels", type=int, default=1, help="핑크노이즈 볼륨 단계 수")
    parser.add_argument(
        "--duration", type=float, default=config.CALIBRATION_MEASURE_SEC
    )
    parser.add_argument("--output", type=Path, default=Path(config.CALIBRATION_FILE))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if arguments.levels < 1:
        parser.error("--levels는 1 이상이어야 합니다")

    server = None
    if arguments.source == "mic":
        device = arguments.device
        if device is None:
            if arguments.mic not in config.MIC_DEVICES:
                parser.error(
                    f"--device가 없고 MIC_DEVICES에 '{arguments.mic}'도 없습니다"
                )
            device = config.MIC_DEVICES[arguments.mic]
        sources, realtime = open_local_source(arguments.mic, device)
        device_text = str(device)
    else:
        sources, realtime, server = open_network_sources(
            arguments.mic, arguments.host, arguments.port
        )
        device_text = "pi"
    monitor = LevelMonitor(sources, realtime_pacing=realtime)
    monitor.start()
    try:
        source = next(source for source in sources if source.name == arguments.mic)
        calibration = run_calibration(
            monitor,
            arguments.mic,
            arguments.levels,
            arguments.duration,
            source.sample_rate,
            device_text,
        )
    finally:
        monitor.stop_event.set()
        for source in sources:
            source.close()
        monitor.join(config.THREAD_JOIN_TIMEOUT_SEC)
        if server is not None:
            server.close()
    report_calibration(arguments.mic, calibration)
    save_mic_calibration(arguments.output, arguments.mic, calibration)
    print(f"저장: {arguments.output}")


if __name__ == "__main__":
    sys.exit(main())
