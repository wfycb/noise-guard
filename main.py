"""파이프라인 조립: 입력 소스 → 레벨·분류(워커 스레드) → 통합 → 판단 → 알림.

실행 예 (프로젝트 루트에서):
    .venv\\Scripts\\python main.py --source mic --mics 거실 --demo
    .venv\\Scripts\\python main.py --source file --files 거실=a.wav,안방=b.wav \\
        --demo --fast --start-time "2026-10-02 23:00" --log-file frames.csv
    .venv\\Scripts\\python main.py --source network --demo
"""

import argparse
import csv
import logging
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy

import config
from alert import AlertSink, ConsoleSink, NetworkSink
from calibration import (
    CalibrationFileError,
    apply_calibration,
    load_calibration,
    warn_uncalibrated,
)
from capture import AudioSource, FileSource, MicSource
from classifier import (
    CedClassifier,
    judgment_limits_db,
    resample_for_classifier,
    validate_skip_gate,
)
from decision import (
    LOCAL_TIMEZONE,
    DecisionConfig,
    NoiseDecisionEngine,
    judge_category,
)
from fusion import fuse_measurements
from level import LevelMeter, leq_db, to_estimated_dba
from mic_health import MicPresenceTracker
from models import Alert, Category, Frame, MicMeasurement
from server_net import AudioServer, ClientSession

START_TIME_FORMAT = "%Y-%m-%d %H:%M"
# argparse는 help 문자열을 % 포맷으로 처리하므로 strftime 형식의 %를 %%로 바꿔 넣는다.
START_TIME_FORMAT_FOR_HELP = START_TIME_FORMAT.replace("%", "%%")
QUEUE_POLL_SEC = 0.5
FRAME_CSV_COLUMNS = [
    "timestamp",
    "local_time",
    "mic",
    "leq_db",
    "lmax_db",
    *(f"prob_{category.value}" for category in Category),
    "top_label",
    "judged",
]
ALERT_CSV_COLUMNS = [
    "timestamp",
    "local_time",
    "level",
    "rule",
    "category",
    "mic",
    "peak_db",
    "count",
    "room_counts",
    "message",
]
ALERT_LOG_SUFFIX = "_alerts"


@dataclass(frozen=True)
class ProducedFrame:
    frame: Frame
    inference_ms: float | None  # 모든 마이크가 조용해 분류를 건너뛰었으면 None
    # 실시간 재생에서 예정 시각(tick) 대비 프레임 완성이 늦은 정도. 누적되면 처리량이 부족한 것이다.
    lag_ms: float | None
    # 이번 tick에 레벨·분류를 낸 마이크. 빠진 마이크 경고에 쓴다.
    present_mics: frozenset[str]
    skipped_mics: int = 0  # 조용해서 CED에 넣지 않은 마이크 수


@dataclass(frozen=True)
class TickWithoutFrame:
    """tick은 지났는데 레벨·분류를 낸 마이크가 하나도 없음(시작 직후 또는 전부 빠짐)."""

    timestamp: float


class ProducerFinished:
    """파일 소스가 모두 끝났거나 --duration에 도달했음을 알리는 큐 표식."""


def parse_mapping(text: str) -> dict[str, str]:
    """ "거실=a.wav,안방=b.wav" → {"거실": "a.wav", "안방": "b.wav"}."""
    mapping = {}
    for item in text.split(","):
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise argparse.ArgumentTypeError(f"'방이름=값' 형식이 아닙니다: {item!r}")
        mapping[name.strip()] = value.strip()
    return mapping


def parse_start_time(text: str) -> float:
    """로컬(Asia/Seoul) "YYYY-MM-DD HH:MM" → epoch 초."""
    local_time = datetime.strptime(text, START_TIME_FORMAT).replace(
        tzinfo=LOCAL_TIMEZONE
    )
    return local_time.timestamp()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="층간소음 경고 시스템")
    parser.add_argument("--source", choices=["mic", "file", "network"], default="mic")
    parser.add_argument(
        "--host", default=config.SERVER_BIND_HOST, help="네트워크 bind 주소"
    )
    parser.add_argument(
        "--port", type=int, default=config.SERVER_PORT, help="네트워크 포트"
    )
    parser.add_argument(
        "--mics", help="사용할 방 이름 (쉼표 구분, config.MIC_DEVICES 중 일부)"
    )
    parser.add_argument(
        "--files", type=parse_mapping, help="방이름=wav경로 (쉼표 구분)"
    )
    parser.add_argument("--demo", action="store_true", help="시연용 짧은 시간창 사용")
    parser.add_argument("--log-file", type=Path, help="프레임별 결과 CSV 경로")
    parser.add_argument(
        "--start-time",
        type=parse_start_time,
        help=f'파일·네트워크 모드 가상 시계 시작 시각 "{START_TIME_FORMAT_FOR_HELP}" (Asia/Seoul)',
    )
    parser.add_argument(
        "--fast", action="store_true", help="파일 모드: 대기 없이 가상 시계로 처리"
    )
    parser.add_argument(
        "--file-end",
        choices=["stop", "pad"],
        default="stop",
        help="파일 모드: 파일이 끝나면 종료(stop)하거나 무음으로 계속(pad)",
    )
    parser.add_argument("--duration", type=float, help="이 시간(초)만큼 처리하고 종료")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=Path(config.CALIBRATION_FILE),
        help="소음계 보정 결과 (tools/calibrate.py가 만든 파일)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="조용한 마이크도 모두 CED로 분류 (SKIP_CLASSIFY_BELOW_DB 게이트 끄기)",
    )
    parser.add_argument(
        "--model",
        choices=list(config.CED_MODELS),
        default=config.CED_MODEL_SIZE,
        help="CED 모델 크기",
    )
    arguments = parser.parse_args()
    if arguments.source == "file" and not arguments.files:
        parser.error("--source file 에는 --files 가 필요합니다")
    if arguments.source == "mic" and (arguments.start_time or arguments.fast):
        parser.error("--start-time/--fast 는 파일·네트워크 모드에서만 쓸 수 있습니다")
    if arguments.source == "network" and arguments.fast:
        parser.error(
            "네트워크 모드는 오디오 도착에 맞춰 처리하므로 --fast가 필요 없습니다"
        )
    return arguments


def build_sources(arguments: argparse.Namespace) -> list[AudioSource]:
    """인자에 맞는 입력 소스를 만든다 (side effect: 장치 열기 또는 파일 읽기)."""
    if arguments.source == "file":
        return [
            FileSource(name, Path(path), arguments.file_end)
            for name, path in arguments.files.items()
        ]
    selected_names = (
        arguments.mics.split(",") if arguments.mics else list(config.MIC_DEVICES)
    )
    unknown_names = set(selected_names) - set(config.MIC_DEVICES)
    if unknown_names:
        raise SystemExit(f"config.MIC_DEVICES에 없는 방: {sorted(unknown_names)}")
    return [MicSource(name, config.MIC_DEVICES[name]) for name in selected_names]


class FrameProducer(threading.Thread):
    """hop마다 모든 소스의 레벨을 계산하고 한 번의 배치로 분류해 Frame을 큐에 넣는다.

    파일 모드에서는 소스에 advance를 호출해 데이터를 공급하고, --fast면 기다리지 않는다.
    프레임 시각은 start_timestamp + tick × hop 이므로 가상 시계로도 판단 로직이 그대로 동작한다.
    """

    def __init__(
        self,
        sources: list[AudioSource],
        classifier: CedClassifier,
        output_queue: queue.Queue,
        start_timestamp: float,
        realtime_pacing: bool,
        stop_when_exhausted: bool,
        duration_sec: float | None,
        tick_ready_time: Callable[[int], float | None] | None = None,
        skip_quiet_below_db: float | None = None,
    ) -> None:
        if skip_quiet_below_db is not None:
            validate_skip_gate(
                skip_quiet_below_db, config.SKIP_MARGIN_DB, judgment_limits_db()
            )
        super().__init__(
            name=f"{config.THREAD_NAME_PREFIX}-frame-producer", daemon=True
        )
        self._sources = sources
        self._classifier = classifier
        self._output_queue = output_queue
        self._start_timestamp = start_timestamp
        self._realtime_pacing = realtime_pacing
        self._stop_when_exhausted = stop_when_exhausted
        self._duration_sec = duration_sec
        # 네트워크 소스: tick 구간 데이터가 모두 도착한 시각. 처리 지연(lag)을 재는 기준이다.
        self._tick_ready_time = tick_ready_time
        # None이면 모든 마이크를 분류한다(--no-skip). 값이 있으면 분류 창(2초) 전체의 Leq가 그보다 낮은
        # 마이크는 건너뛴다. 1초 Leq로 보면, 큰 소리가 막 끝난 프레임(창 안에는 그 소리가 있어 CED가
        # IMPACT로 보는 프레임)까지 건너뛰어 이벤트 병합이 달라지고 알림이 --no-skip과 달라진다.
        self._skip_quiet_below_db = skip_quiet_below_db
        self._level_meters = {
            source.name: LevelMeter(source.sample_rate) for source in sources
        }
        window_block_count = round(config.CLASSIFY_WINDOW_SEC / config.FAST_BLOCK_SEC)
        self._window_block_levels = {
            source.name: deque(maxlen=window_block_count) for source in sources
        }
        self._total_read = {source.name: 0 for source in sources}
        self.stop_event = threading.Event()

    def run(self) -> None:
        # 워커에서 난 예외를 메인 스레드로 넘겨야 조용히 멈추지 않는다.
        try:
            self._produce_frames()
        except Exception as error:  # noqa: BLE001 — 메인 스레드에서 다시 raise한다
            self._output_queue.put(error)
        else:
            self._output_queue.put(ProducerFinished())

    def _produce_frames(self) -> None:
        hop_sec = config.CLASSIFY_HOP_SEC
        pacing_start = time.monotonic()
        tick = 0
        while not self.stop_event.is_set():
            if self._stop_when_exhausted and all(
                source.exhausted for source in self._sources
            ):
                return
            if self._duration_sec is not None and tick * hop_sec >= self._duration_sec:
                return
            tick += 1
            if self._realtime_pacing:
                wait_sec = pacing_start + tick * hop_sec - time.monotonic()
                if self.stop_event.wait(max(0.0, wait_sec)):
                    return
            for source in self._sources:
                source.advance(hop_sec)
            produced = self._measure_and_classify(
                self._start_timestamp + tick * hop_sec
            )
            timestamp = self._start_timestamp + tick * hop_sec
            if produced is None:
                # 소스가 모두 끝나서 데이터가 없는 tick은 마이크 빠짐이 아니라 종료다.
                if self._stop_when_exhausted and all(
                    source.exhausted for source in self._sources
                ):
                    return
                # 분류 창이 차기 전 tick은 어떤 마이크도 프레임을 낼 수 없으므로 빠짐으로 세지 않는다.
                if tick * hop_sec >= config.CLASSIFY_WINDOW_SEC:
                    self._output_queue.put(TickWithoutFrame(timestamp))
                continue
            lag_ms = self._lag_ms(tick, pacing_start)
            if lag_ms is not None:
                produced = ProducedFrame(
                    produced.frame,
                    produced.inference_ms,
                    lag_ms,
                    produced.present_mics,
                    produced.skipped_mics,
                )
            self._output_queue.put(produced)

    def _lag_ms(self, tick: int, pacing_start: float) -> float | None:
        """tick 예정 시각(실시간 재생) 또는 데이터 도착 시각(네트워크) 대비 완성 지연."""
        if self._realtime_pacing:
            return (
                time.monotonic() - pacing_start - tick * config.CLASSIFY_HOP_SEC
            ) * 1000.0
        if self._tick_ready_time is not None:
            ready_time = self._tick_ready_time(tick)
            if ready_time is not None:
                return (time.perf_counter() - ready_time) * 1000.0
        return None

    def _window_leq_db(self, name: str) -> float:
        """분류 창(CLASSIFY_WINDOW_SEC)과 같은 구간의 Leq, 추정 dB(A)."""
        return to_estimated_dba(leq_db(list(self._window_block_levels[name])), name)

    def _measure_and_classify(self, timestamp: float) -> ProducedFrame | None:
        """소스마다 새 샘플로 레벨을, 최근 창으로 분류를 계산해 Frame 하나로 합친다."""
        block_levels_by_name: dict[str, list[float]] = {}
        windows: dict[str, numpy.ndarray] = {}
        for source in self._sources:
            new_samples, self._total_read[source.name] = source.read_new(
                self._total_read[source.name]
            )
            block_levels = self._level_meters[source.name].push(new_samples)
            self._window_block_levels[source.name].extend(block_levels)
            window = source.latest(
                round(config.CLASSIFY_WINDOW_SEC * source.sample_rate)
            )
            # 시작 직후에는 분류 창(2초)이 아직 안 찼으므로 그 마이크는 건너뛴다.
            if block_levels and window is not None:
                block_levels_by_name[source.name] = block_levels
                windows[source.name] = resample_for_classifier(
                    window, source.sample_rate
                )
        if not windows:
            return None

        leq_by_name = {
            name: to_estimated_dba(leq_db(levels), name)
            for name, levels in block_levels_by_name.items()
        }
        # 조용한 마이크는 분류하지 않는다. "크면 분류 없이 통과"는 하지 않는다(물소리를 못 거르므로).
        loud_windows = {
            name: window
            for name, window in windows.items()
            if self._skip_quiet_below_db is None
            or self._window_leq_db(name) >= self._skip_quiet_below_db
        }
        inference_ms = None
        results = {}
        if loud_windows:
            inference_start = time.perf_counter()
            results = self._classifier.classify(loud_windows)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

        quiet_probs = {category: 0.0 for category in Category}
        measurements = {}
        for name, levels in block_levels_by_name.items():
            result = results.get(name)
            measurements[name] = MicMeasurement(
                leq_db=leq_by_name[name],
                lmax_db=to_estimated_dba(max(levels), name),
                category_probs=result.category_probs if result else quiet_probs,
                top_label=result.top_labels[0][0] if result else config.QUIET_TOP_LABEL,
            )
        return ProducedFrame(
            fuse_measurements(timestamp, measurements),
            inference_ms,
            lag_ms=None,
            present_mics=frozenset(measurements),
            skipped_mics=len(windows) - len(loud_windows),
        )


def format_local_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, LOCAL_TIMEZONE).isoformat(
        timespec="seconds"
    )


class CsvRunLogger:
    """프레임별 결과와 원본 Alert 전체를 CSV 두 개로 저장한다 (side effect: 파일 쓰기).

    알림은 콘솔에서는 프레임 단위로 합쳐 보이지만, 분석용 CSV에는 규칙별 원본을 그대로 남긴다.
    알림 CSV 경로는 프레임 CSV 이름에 "_alerts"를 붙인 것이다.
    """

    def __init__(self, frame_log_path: Path) -> None:
        alert_log_path = frame_log_path.with_name(
            f"{frame_log_path.stem}{ALERT_LOG_SUFFIX}{frame_log_path.suffix}"
        )
        self._frame_file: TextIO = frame_log_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._alert_file: TextIO = alert_log_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._frame_writer = csv.writer(self._frame_file)
        self._alert_writer = csv.writer(self._alert_file)
        self._frame_writer.writerow(FRAME_CSV_COLUMNS)
        self._alert_writer.writerow(ALERT_CSV_COLUMNS)

    def write_frame(self, frame: Frame, judged: Category | None) -> None:
        self._frame_writer.writerow(
            [
                f"{frame.timestamp:.3f}",
                format_local_iso(frame.timestamp),
                frame.mic_name,
                f"{frame.leq_db:.2f}",
                f"{frame.lmax_db:.2f}",
                *(
                    f"{frame.category_probs.get(category, 0.0):.4f}"
                    for category in Category
                ),
                frame.top_label,
                judged.value if judged else "",
            ]
        )
        # 중간에 Ctrl+C로 끊겨도 그때까지의 기록은 남게 한다.
        self._frame_file.flush()

    def write_alerts(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            self._alert_writer.writerow(
                [
                    f"{alert.timestamp:.3f}",
                    format_local_iso(alert.timestamp),
                    alert.level,
                    alert.rule,
                    alert.category.value,
                    alert.mic_name,
                    f"{alert.peak_db:.2f}",
                    alert.count,
                    ";".join(
                        f"{room}={count}" for room, count in alert.room_counts.items()
                    ),
                    alert.message,
                ]
            )
        self._alert_file.flush()

    def close(self) -> None:
        self._frame_file.close()
        self._alert_file.close()


@dataclass
class RunResult:
    frames: list[Frame] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    lag_ms: list[float] = field(default_factory=list)
    overflow_count: int = 0
    # Ctrl+C로 끝났는지 (네트워크 모드에서 다음 연결을 기다릴지 결정)
    interrupted: bool = False
    # 마이크별로 tick에서 빠진 횟수 (분류 창이 차기 전 tick은 제외)
    missing_ticks: dict[str, int] = field(default_factory=dict)
    mic_frames: int = 0  # 레벨을 낸 마이크-프레임 수
    skipped_mic_frames: int = 0  # 그중 조용해서 분류를 건너뛴 수
    overrun_ticks: int = (
        0  # 처리 지연이 hop을 넘은 tick 수 (실시간이 밀리기 시작한 신호)
    )


def run_sources(
    sources: list[AudioSource],
    classifier: CedClassifier,
    engine: NoiseDecisionEngine,
    producer_options: dict[str, Any],
    notifier: AlertSink | list[AlertSink] | None,
    logger: CsvRunLogger | None,
) -> RunResult:
    """소스를 시작하고 워커가 만든 프레임을 판단·출력·기록한다 (side effect: 장치/파일/출력).

    Ctrl+C를 받으면 그때까지의 결과를 반환한다. producer_options는 FrameProducer 인자
    (start_timestamp, realtime_pacing, stop_when_exhausted, duration_sec, tick_ready_time)다.
    notifier는 출력 하나 또는 여러 개(예: 콘솔 + 네트워크)다.
    """
    if notifier is None:
        sinks: list[AlertSink] = []
    elif isinstance(notifier, list):
        sinks = notifier
    else:
        sinks = [notifier]
    frame_queue: queue.Queue = queue.Queue()
    producer = FrameProducer(sources, classifier, frame_queue, **producer_options)
    result = RunResult()
    tracker = MicPresenceTracker(
        [source.name for source in sources],
        config.MIC_MISSING_WARN_TICKS,
        config.CLASSIFY_HOP_SEC,
    )
    for source in sources:
        source.start()
    producer.start()
    try:
        while True:
            try:
                item = frame_queue.get(timeout=QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            if isinstance(item, ProducerFinished):
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, TickWithoutFrame):
                report_mic_status(tracker, item.timestamp, frozenset(), sinks)
                continue
            frame = item.frame
            report_mic_status(tracker, frame.timestamp, item.present_mics, sinks)
            result.frames.append(frame)
            if item.inference_ms is not None:
                result.inference_ms.append(item.inference_ms)
            result.mic_frames += len(item.present_mics)
            result.skipped_mic_frames += item.skipped_mics
            if item.lag_ms is not None:
                result.lag_ms.append(item.lag_ms)
                report_processing_delay(item.lag_ms, result)
            judged = judge_category(frame.category_probs, config.CLASS_PROB_THRESHOLD)
            alerts = engine.update(frame)
            result.alerts.extend(alerts)
            for sink in sinks:
                sink.emit_frame(
                    frame, judged, tracker.last_active_count, tracker.total_mics
                )
                sink.emit(alerts, frame)
            if logger:
                logger.write_frame(frame, judged)
                logger.write_alerts(alerts)
    except KeyboardInterrupt:
        result.interrupted = True
    finally:
        producer.stop_event.set()
        # 네트워크 소스는 advance 안에서 데이터를 기다리므로, 소스를 먼저 닫아 깨운 뒤 join한다.
        for source in sources:
            source.close()
        producer.join(config.THREAD_JOIN_TIMEOUT_SEC)
        result.overflow_count = sum(source.overflow_count for source in sources)
        result.missing_ticks = tracker.missing_tick_counts()
    return result


def report_processing_delay(lag_ms: float, result: RunResult) -> None:
    """처리 지연이 hop을 넘으면 경고 로그를 남기고 센다 (side effect: 로그).

    hop보다 오래 걸리는 tick이 이어지면 처리가 실시간을 따라가지 못해 지연이 쌓인다.
    """
    hop_ms = config.CLASSIFY_HOP_SEC * 1000.0
    if lag_ms <= hop_ms:
        return
    result.overrun_ticks += 1
    logging.getLogger(__name__).warning("[처리 지연] %.0f ms > %.0f ms", lag_ms, hop_ms)


def report_mic_status(
    tracker: MicPresenceTracker,
    timestamp: float,
    present_mics: frozenset[str],
    sinks: list[AlertSink],
) -> None:
    """tick 하나의 마이크 참여를 반영하고, 상태가 바뀌었으면 출력한다 (side effect: 출력)."""
    change = tracker.update(timestamp, set(present_mics))
    if change is None:
        return
    for sink in sinks:
        sink.emit_mic_status(change)


@dataclass
class SessionRunResult:
    result: RunResult
    session: ClientSession
    network_sink: NetworkSink


def serve_network(
    server: AudioServer,
    classifier: CedClassifier,
    engine: NoiseDecisionEngine,
    notifier: AlertSink | None,
    logger: CsvRunLogger | None,
    max_sessions: int | None = None,
    skip_quiet_below_db: float | None = None,
) -> list[SessionRunResult]:
    """연결을 하나씩 받아 처리한다 (side effect: 소켓·출력). 엔진은 연결 사이에 유지한다.

    max_sessions에 도달하거나 Ctrl+C를 받으면 끝난다. 연결 대기 중 Ctrl+C는 호출자에게 전달된다.
    """
    session_results: list[SessionRunResult] = []
    last_frame_timestamp: float | None = None
    while max_sessions is None or len(session_results) < max_sessions:
        session = server.next_session(timeout=config.SERVER_THREAD_POLL_SEC)
        if session is None:
            continue
        # 판단 엔진은 시각이 앞으로만 간다고 가정한다. 재접속한 세션의 시작 시각이 이전 세션의
        # 마지막 프레임보다 앞서면(고속 재생 등) 그 뒤로 옮긴다. 실시간이면 일어나지 않는다.
        if last_frame_timestamp is not None and (
            session.stream_start_time < last_frame_timestamp
        ):
            logging.getLogger(__name__).warning(
                "세션 시작 시각을 이전 세션 뒤로 %.1f초 옮깁니다",
                last_frame_timestamp - session.stream_start_time,
            )
            session.stream_start_time = last_frame_timestamp
        network_sink = NetworkSink(session, config.CLASSIFY_HOP_SEC)
        warn_uncalibrated([source.name for source in session.sources])
        sinks: list[AlertSink] = (
            [network_sink] if notifier is None else [notifier, network_sink]
        )
        result = run_sources(
            session.sources,
            classifier,
            engine,
            producer_options={
                "start_timestamp": session.stream_start_time,
                "realtime_pacing": False,
                "stop_when_exhausted": True,
                "duration_sec": None,
                "tick_ready_time": session.stream.tick_ready_time,
                "skip_quiet_below_db": skip_quiet_below_db,
            },
            notifier=sinks,
            logger=logger,
        )
        session.close("서버 중단" if result.interrupted else None)
        session.join()
        logging.getLogger(__name__).info(
            "%s: 세션 종료 (%s), 프레임 %d",
            session.address,
            session.close_reason,
            len(result.frames),
        )
        session_results.append(SessionRunResult(result, session, network_sink))
        if result.frames:
            last_frame_timestamp = result.frames[-1].timestamp
        if result.interrupted:
            break
    return session_results


def print_summary(result: RunResult) -> None:
    """종료 요약을 출력한다 (side effect: 콘솔 출력)."""
    alerts = result.alerts
    rule_counts = Counter(alert.rule for alert in alerts)
    print("\n===== 요약 =====")
    print(f"총 프레임: {len(result.frames)}")
    print(
        f"알림(규칙별 원본): {len(alerts)}건 "
        f"(caution {sum(a.level == 'caution' for a in alerts)}, "
        f"warning {sum(a.level == 'warning' for a in alerts)}) "
        f"규칙별 {dict(sorted(rule_counts.items()))}"
    )
    print(f"overflow: {result.overflow_count}")
    if result.mic_frames:
        print(
            f"분류 건너뜀(조용함): {result.skipped_mic_frames}/{result.mic_frames} "
            f"마이크-프레임 ({100 * result.skipped_mic_frames / result.mic_frames:.0f}%)"
        )
    if result.missing_ticks:
        missing_text = ", ".join(
            f"{name} {count}" for name, count in result.missing_ticks.items()
        )
        print(f"마이크별 빠진 tick: {missing_text}")
    if result.inference_ms:
        print(
            f"배치 추론: 평균 {numpy.mean(result.inference_ms):.1f} ms, "
            f"최대 {numpy.max(result.inference_ms):.1f} ms"
        )
    if result.lag_ms:
        print(
            f"처리 지연: 평균 {numpy.mean(result.lag_ms):.1f} ms, "
            f"최대 {numpy.max(result.lag_ms):.1f} ms, "
            f"마지막 {result.lag_ms[-1]:.1f} ms, "
            f"hop({config.CLASSIFY_HOP_SEC * 1000:.0f} ms) 초과 {result.overrun_ticks}회"
        )


def print_network_stats(session: ClientSession) -> None:
    """세션의 마이크별 수신 통계를 출력한다 (side effect: 콘솔 출력)."""
    print(f"연결 {session.address}: {session.close_reason}")
    for room, stats in session.stats().items():
        print(
            f"  {room}: 무음 채움 {stats.gap_filled_chunks}청크, "
            f"늦게 와서 버림 {stats.late_discarded_samples}샘플, "
            f"마감 초과 제외 {stats.stalled_ticks}tick, "
            f"Pi overflow {stats.overflow_chunks}청크"
        )


def skip_gate_db(arguments: argparse.Namespace) -> float | None:
    """--no-skip이면 None(모두 분류), 아니면 config의 게이트 값."""
    return None if arguments.no_skip else config.SKIP_CLASSIFY_BELOW_DB


def run_network_pipeline(
    arguments: argparse.Namespace,
    classifier: CedClassifier,
    engine: NoiseDecisionEngine,
    logger: CsvRunLogger | None,
) -> None:
    """서버를 열고 Ctrl+C까지 연결을 처리한다 (side effect: 소켓·출력)."""
    server = AudioServer(
        arguments.host, arguments.port, start_time=arguments.start_time
    )
    server.start()
    session_results: list[SessionRunResult] = []
    try:
        session_results = serve_network(
            server,
            classifier,
            engine,
            ConsoleSink(),
            logger,
            skip_quiet_below_db=skip_gate_db(arguments),
        )
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    for session_result in session_results:
        print_summary(session_result.result)
        print_network_stats(session_result.session)


def load_and_apply_calibration(path: Path) -> None:
    """보정 파일이 있으면 방별 오프셋을 적용한다 (side effect: 파일 읽기, config 변경, 로그)."""
    try:
        calibrations = load_calibration(path)
    except CalibrationFileError as error:
        raise SystemExit(f"보정 파일 오류: {error}") from error
    apply_calibration(calibrations)
    if calibrations:
        logging.getLogger(__name__).info(
            "보정 적용 (%s): %s",
            path,
            ", ".join(
                f"{room} {value.offset_db:+.1f} dB"
                for room, value in calibrations.items()
            ),
        )
    else:
        logging.getLogger(__name__).warning(
            "보정 파일 %s가 없습니다. 모든 방에 임시 오프셋 +%.0f dB를 씁니다",
            path,
            config.DEFAULT_CALIBRATION_OFFSET_DB,
        )


def run_pipeline(arguments: argparse.Namespace) -> None:
    """인자로 소스·엔진을 만들고 실행한 뒤 요약을 출력한다 (side effect: 장치/파일/출력)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_and_apply_calibration(arguments.calibration_file)
    demo_mode = arguments.demo or config.DEMO_MODE
    engine = NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=demo_mode))
    print("CED 로딩 중...")
    classifier = CedClassifier(model_size=arguments.model)
    print(
        f"모델: {classifier.model_name} | 조용할 때 분류 건너뛰기: "
        + (
            "끔"
            if arguments.no_skip
            else f"Leq < {config.SKIP_CLASSIFY_BELOW_DB} dB(A) (보정 전 임시 기준)"
        )
    )
    if arguments.source == "network":
        logger = CsvRunLogger(arguments.log_file) if arguments.log_file else None
        try:
            run_network_pipeline(arguments, classifier, engine, logger)
        finally:
            if logger:
                logger.close()
        return
    sources = build_sources(arguments)
    print(
        f"소스: {', '.join(f'{s.name}({s.sample_rate}Hz)' for s in sources)} | "
        f"demo={demo_mode} | scope={config.AIRBORNE_SCOPE}"
    )
    warn_uncalibrated([source.name for source in sources])
    logger = CsvRunLogger(arguments.log_file) if arguments.log_file else None
    try:
        result = run_sources(
            sources,
            classifier,
            engine,
            producer_options={
                "start_timestamp": arguments.start_time or time.time(),
                "realtime_pacing": not arguments.fast,
                "stop_when_exhausted": arguments.source == "file"
                and arguments.file_end == "stop",
                "duration_sec": arguments.duration,
                "skip_quiet_below_db": skip_gate_db(arguments),
            },
            notifier=ConsoleSink(),
            logger=logger,
        )
    finally:
        if logger:
            logger.close()
    print_summary(result)


if __name__ == "__main__":
    run_pipeline(parse_arguments())
