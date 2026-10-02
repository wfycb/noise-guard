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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy

import config
from alert import AlertSink, ConsoleSink, NetworkSink
from capture import AudioSource, FileSource, MicSource
from classifier import CedClassifier, resample_for_classifier
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
    inference_ms: float
    # 실시간 재생에서 예정 시각(tick) 대비 프레임 완성이 늦은 정도. 누적되면 처리량이 부족한 것이다.
    lag_ms: float | None
    # 이번 tick에 레벨·분류를 낸 마이크. 빠진 마이크 경고에 쓴다.
    present_mics: frozenset[str]


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
        help=f'파일·네트워크 모드 가상 시계 시작 시각 "{START_TIME_FORMAT}" (Asia/Seoul)',
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
    ) -> None:
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
        self._level_meters = {
            source.name: LevelMeter(source.sample_rate) for source in sources
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
                self._output_queue.put(TickWithoutFrame(timestamp))
                continue
            lag_ms = self._lag_ms(tick, pacing_start)
            if lag_ms is not None:
                produced = ProducedFrame(
                    produced.frame, produced.inference_ms, lag_ms, produced.present_mics
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

    def _measure_and_classify(self, timestamp: float) -> ProducedFrame | None:
        """소스마다 새 샘플로 레벨을, 최근 창으로 분류를 계산해 Frame 하나로 합친다."""
        block_levels_by_name: dict[str, list[float]] = {}
        windows: dict[str, numpy.ndarray] = {}
        for source in self._sources:
            new_samples, self._total_read[source.name] = source.read_new(
                self._total_read[source.name]
            )
            block_levels = self._level_meters[source.name].push(new_samples)
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

        inference_start = time.perf_counter()
        results = self._classifier.classify(windows)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        measurements = {
            name: MicMeasurement(
                leq_db=to_estimated_dba(leq_db(block_levels_by_name[name]), name),
                lmax_db=to_estimated_dba(max(block_levels_by_name[name]), name),
                category_probs=result.category_probs,
                top_label=result.top_labels[0][0],
            )
            for name, result in results.items()
        }
        return ProducedFrame(
            fuse_measurements(timestamp, measurements),
            inference_ms,
            lag_ms=None,
            present_mics=frozenset(measurements),
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
    # 마이크별로 tick에서 빠진 횟수 (시작 직후 분류 창이 차기 전 tick 포함)
    missing_ticks: dict[str, int] = field(default_factory=dict)


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
            result.inference_ms.append(item.inference_ms)
            if item.lag_ms is not None:
                result.lag_ms.append(item.lag_ms)
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
) -> list[SessionRunResult]:
    """연결을 하나씩 받아 처리한다 (side effect: 소켓·출력). 엔진은 연결 사이에 유지한다.

    max_sessions에 도달하거나 Ctrl+C를 받으면 끝난다. 연결 대기 중 Ctrl+C는 호출자에게 전달된다.
    """
    session_results: list[SessionRunResult] = []
    while max_sessions is None or len(session_results) < max_sessions:
        session = server.next_session(timeout=config.SERVER_THREAD_POLL_SEC)
        if session is None:
            continue
        network_sink = NetworkSink(session, config.CLASSIFY_HOP_SEC)
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
            f"실시간 지연(예정 시각 대비): 평균 {numpy.mean(result.lag_ms):.1f} ms, "
            f"최대 {numpy.max(result.lag_ms):.1f} ms, "
            f"마지막 {result.lag_ms[-1]:.1f} ms"
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
            server, classifier, engine, ConsoleSink(), logger
        )
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    for session_result in session_results:
        print_summary(session_result.result)
        print_network_stats(session_result.session)


def run_pipeline(arguments: argparse.Namespace) -> None:
    """인자로 소스·엔진을 만들고 실행한 뒤 요약을 출력한다 (side effect: 장치/파일/출력)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    demo_mode = arguments.demo or config.DEMO_MODE
    engine = NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=demo_mode))
    print("CED 로딩 중...")
    classifier = CedClassifier()
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
        f"demo={demo_mode} | scope={config.AIRBORNE_SCOPE} | "
        f"dB(A)는 보정 전 임시 오프셋 적용값"
    )
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
