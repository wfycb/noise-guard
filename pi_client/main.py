"""Pi 진입점: 수집한 오디오를 서버로 보내고, 서버 명령을 받는다 (side effect: 장치·소켓·스레드).

실행 (저장소 루트에서):
    python -m pi_client.main --server 192.168.10.1 --port 5000 --source mic
    python -m pi_client.main --server 127.0.0.1 --source file \\
        --files 거실=a.wav,안방=b.wav --fast

오류 정책 (protocol.py의 예외 구분):
- FramingError(경계 깨짐), 서버 연결 끊김 → 연결을 닫는다.
- PayloadError(ALERT·STATUS 내용 오류) → 경고 로그를 남기고 그 메시지만 무시한다. 오디오 전송은 계속한다.

종단 지연: ALERT의 (source_mic_index, source_seq)로, 그 청크를 보낼 때 기록한 캡처 시각을 찾아
"수신 시각 − 캡처 시각"을 로그에 남긴다. 두 값 모두 이 Pi의 time.perf_counter()라 시계 동기화가 필요 없다.
"""

import argparse
import logging
import socket
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Protocol

from pi_client import client_config
from pi_client.capture import (
    ChunkQueue,
    FileStream,
    MicStream,
    StallSpec,
    parse_stall_spec,
)
from pi_client.outputs import ConsoleOutput, HardwareOutput, OutputController
from protocol import (
    PROTOCOL_VERSION,
    AudioChunk,
    ConnectionClosed,
    FramingError,
    Hello,
    MessageType,
    MicInfo,
    PayloadError,
    ProtocolError,
    decode_alert,
    decode_hello_ack,
    decode_status,
    encode_audio,
    encode_hello,
    read_message,
    send_message,
    validate_ping_interval,
)

logger = logging.getLogger("pi_client")

EXIT_OK = 0
EXIT_REJECTED = 2
EXIT_CONNECTION_FAILED = 3


class ChunkSource(Protocol):
    finished: threading.Event

    @property
    def mics(self) -> tuple[MicInfo, ...]: ...

    @property
    def overflow_count(self) -> int: ...

    def start(self, chunk_queue: ChunkQueue, stop_event: threading.Event) -> None: ...

    def reset_sequence(self) -> None: ...

    def stop(self) -> None: ...


class SessionOutcome(Enum):
    DONE = "보낼 것을 다 보내고 정상 종료"
    STOPPED = "종료 요청"
    LOST = "연결 끊김"
    CONNECT_FAILED = "연결 실패"
    REJECTED = "서버가 거부"


class PiClient:
    """서버 연결을 유지하며 오디오를 보내고 명령을 받는다.

    연결 하나는 _run_session이 맡는다(HELLO → 수신 스레드 → 송신 루프). 끊기면 reconnect일 때
    지수 백오프(최대 reconnect_max_sec)로 다시 연결한다. 재접속하면 seq를 0부터 다시 세고,
    끊긴 동안 쌓인 오디오는 버린다. 수집(source)은 처음 연결된 뒤 계속 돌린다.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: str,
        source: ChunkSource,
        chunk_samples: int,
        output: HardwareOutput | None = None,
        display_hold_sec: float = client_config.DISPLAY_HOLD_SEC,
        reconnect: bool = False,
        reconnect_initial_sec: float = client_config.RECONNECT_INITIAL_SEC,
        reconnect_max_sec: float = client_config.RECONNECT_MAX_SEC,
        ping_interval_sec: float = client_config.PING_INTERVAL_SEC,
        peer_timeout_sec: float = client_config.PEER_TIMEOUT_SEC,
    ) -> None:
        validate_ping_interval(ping_interval_sec, peer_timeout_sec)
        self._host = host
        self._port = port
        self._client_id = client_id
        self._source = source
        self._chunk_samples = chunk_samples
        self._reconnect = reconnect
        self._reconnect_initial_sec = reconnect_initial_sec
        self._reconnect_max_sec = reconnect_max_sec
        self._ping_interval_sec = ping_interval_sec
        self._peer_timeout_sec = peer_timeout_sec
        self._queue = ChunkQueue(client_config.SEND_QUEUE_MAX_CHUNKS)
        self._stop = threading.Event()
        self._link_lost = threading.Event()
        self._finishing = False
        self._send_lock = threading.Lock()
        self._outputs = OutputController(output or ConsoleOutput(), display_hold_sec)
        self.sent_chunks = 0
        self.received_alerts: list[dict[str, object]] = []
        self.received_statuses: list[object] = []
        self.ignored_payloads = 0
        self.connections = 0
        self.connections_before_attempt = 0
        self.discarded_on_reconnect = 0
        self.lost_reasons: list[str] = []
        # 종단 지연 측정용: (mic_index, seq) → 캡처 시각. 최근 LATENCY_HISTORY_SEC만 기억한다.
        self._capture_times: dict[tuple[int, int], float] = {}
        self._capture_order: deque[tuple[float, tuple[int, int]]] = deque()
        self._capture_lock = threading.Lock()
        self.latencies_ms: list[float] = []
        self.unmeasured_alerts = 0

    @property
    def dropped_chunks(self) -> int:
        return self._queue.dropped_chunks

    def stop(self) -> None:
        """다른 스레드에서 종료를 요청한다."""
        self._stop.set()

    def run(self) -> int:
        """끝날 때까지 연결·전송·재접속을 반복한다 (side effect: 소켓·장치). 종료 코드를 반환한다."""
        self._outputs.start()
        backoff_sec = self._reconnect_initial_sec
        disconnected_at: float | None = None
        dropped_at_disconnect = 0
        outcome = SessionOutcome.STOPPED
        try:
            while not self._stop.is_set():
                outcome = self._run_session(disconnected_at, dropped_at_disconnect)
                if outcome in (SessionOutcome.DONE, SessionOutcome.STOPPED):
                    break
                if self._stop.is_set() or not self._reconnect:
                    break
                if self.connections and disconnected_at is None:
                    disconnected_at = time.perf_counter()
                    dropped_at_disconnect = self._queue.dropped_chunks
                if self.connections_before_attempt != self.connections:
                    backoff_sec = self._reconnect_initial_sec
                logger.warning(
                    "%s — %.1f초 후 다시 연결합니다", outcome.value, backoff_sec
                )
                if self._stop.wait(backoff_sec):
                    break
                backoff_sec = min(backoff_sec * 2, self._reconnect_max_sec)
        finally:
            self._stop.set()
            self._source.stop()
            self._outputs.stop()
        logger.info(
            "종료: 연결 %d회, 보낸 청크 %d, 큐에서 버린 청크 %d, 재접속 시 버린 청크 %d, "
            "overflow %d, 받은 알림 %d, 종단 지연 측정 %d건(측정 불가 %d건)",
            self.connections,
            self.sent_chunks,
            self.dropped_chunks,
            self.discarded_on_reconnect,
            self._source.overflow_count,
            len(self.received_alerts),
            len(self.latencies_ms),
            self.unmeasured_alerts,
        )
        if outcome == SessionOutcome.REJECTED:
            return EXIT_REJECTED
        if outcome in (SessionOutcome.CONNECT_FAILED, SessionOutcome.LOST):
            return EXIT_CONNECTION_FAILED
        return EXIT_OK

    def _run_session(
        self, disconnected_at: float | None, dropped_at_disconnect: int
    ) -> SessionOutcome:
        """연결 하나를 처음부터 끝까지 처리한다 (side effect: 소켓·스레드)."""
        self.connections_before_attempt = self.connections
        try:
            connection = socket.create_connection(
                (self._host, self._port), timeout=client_config.CONNECT_TIMEOUT_SEC
            )
        except OSError as error:
            logger.error("서버 %s:%d 연결 실패 — %s", self._host, self._port, error)
            return SessionOutcome.CONNECT_FAILED
        receiver: threading.Thread | None = None
        try:
            try:
                accepted = self._handshake(connection)
            except (OSError, ConnectionClosed, ProtocolError) as error:
                logger.error("HELLO 교환 실패 — %s", error)
                return SessionOutcome.CONNECT_FAILED
            if not accepted:
                return SessionOutcome.REJECTED
            self.connections += 1
            if disconnected_at is not None:
                self._report_reconnect(disconnected_at, dropped_at_disconnect)
            self._link_lost.clear()
            self._finishing = False
            # 수신 타임아웃: PING이 오가므로 이 시간 동안 아무것도 못 받으면 끊긴 것이다.
            connection.settimeout(self._peer_timeout_sec)
            receiver = threading.Thread(
                target=self._receive_loop,
                args=(connection,),
                name=f"{client_config.THREAD_NAME_PREFIX}-recv",
                daemon=True,
            )
            receiver.start()
            if self.connections == 1:
                self._source.start(self._queue, self._stop)
            result = self._send_loop(connection)
            if result == SessionOutcome.DONE:
                self._finish(connection, receiver)
            return result
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # 이미 끊긴 경우
            connection.close()
            if receiver is not None:
                receiver.join(client_config.THREAD_JOIN_TIMEOUT_SEC)

    def _report_reconnect(
        self, disconnected_at: float, dropped_at_disconnect: int
    ) -> None:
        """재접속: seq를 처음부터 세고, 끊긴 동안 쌓인 오디오를 버린다 (side effect: 로그)."""
        self._source.reset_sequence()
        cleared = self._queue.clear()
        overflowed = self._queue.dropped_chunks - dropped_at_disconnect
        self.discarded_on_reconnect += cleared + overflowed
        logger.warning(
            "재접속 성공: 끊긴 시간 %.1f초, 버린 청크 %d (큐 넘침 %d + 재접속 시 비움 %d)",
            time.perf_counter() - disconnected_at,
            cleared + overflowed,
            overflowed,
            cleared,
        )

    def _handshake(self, connection: socket.socket) -> bool:
        hello = Hello(
            PROTOCOL_VERSION, self._client_id, self._source.mics, self._chunk_samples
        )
        connection.settimeout(client_config.HELLO_ACK_TIMEOUT_SEC)
        send_message(connection, MessageType.HELLO, encode_hello(hello))
        message = read_message(connection)
        if message.message_type != MessageType.HELLO_ACK:
            logger.error("HELLO_ACK 대신 %s를 받았습니다", message.message_type.name)
            return False
        ack = decode_hello_ack(message.body)
        if not ack.accepted:
            logger.error("서버가 연결을 거부했습니다: %s", "; ".join(ack.reasons))
            return False
        logger.info(
            "연결됨 %s:%d, 마이크 %s",
            self._host,
            self._port,
            [(mic.room, mic.sample_rate) for mic in hello.mics],
        )
        return True

    def _send(
        self, connection: socket.socket, message_type: MessageType, body: bytes
    ) -> None:
        # 수신 스레드(PONG)와 송신 루프(AUDIO, PING)가 같은 소켓에 쓰므로 메시지 단위로 잠근다.
        with self._send_lock:
            send_message(connection, message_type, body)

    def _send_loop(self, connection: socket.socket) -> SessionOutcome:
        last_ping = time.perf_counter()
        while not self._stop.is_set():
            if self._link_lost.is_set():
                return SessionOutcome.LOST
            try:
                if time.perf_counter() - last_ping >= self._ping_interval_sec:
                    self._send(connection, MessageType.PING, b"")
                    last_ping = time.perf_counter()
                item = self._queue.get(timeout=client_config.QUEUE_POLL_SEC)
                if item is None:
                    if self._source.finished.is_set() and len(self._queue) == 0:
                        return SessionOutcome.DONE
                    continue
                body = encode_audio(
                    AudioChunk(item.mic_index, item.seq, item.flags, item.samples)
                )
                self._send(connection, MessageType.AUDIO, body)
            except OSError as error:
                self._mark_lost(f"송신 실패: {error}")
                return SessionOutcome.LOST
            self.sent_chunks += 1
            self._remember_capture_time(item.mic_index, item.seq, item.capture_time)
        return SessionOutcome.STOPPED

    def _mark_lost(self, reason: str) -> None:
        if not self._link_lost.is_set():
            self.lost_reasons.append(reason)
            logger.warning("연결 끊김 — %s", reason)
        self._link_lost.set()

    def _remember_capture_time(
        self, mic_index: int, seq: int, capture_time: float
    ) -> None:
        with self._capture_lock:
            key = (mic_index, seq)
            self._capture_times[key] = capture_time
            self._capture_order.append((capture_time, key))
            cutoff = time.perf_counter() - client_config.LATENCY_HISTORY_SEC
            while self._capture_order and self._capture_order[0][0] < cutoff:
                _, old_key = self._capture_order.popleft()
                self._capture_times.pop(old_key, None)

    def _record_latency(self, alert: dict[str, object]) -> None:
        """ALERT가 가리키는 청크의 캡처 시각부터 지금까지를 종단 지연으로 기록한다 (side effect: 로그)."""
        key = (int(alert["source_mic_index"]), int(alert["source_seq"]))
        with self._capture_lock:
            capture_time = self._capture_times.get(key)
        if capture_time is None:
            self.unmeasured_alerts += 1
            logger.info(
                "종단 지연 측정 불가: mic %d seq %d가 최근 %g초 기록 밖",
                key[0],
                key[1],
                client_config.LATENCY_HISTORY_SEC,
            )
            return
        latency_ms = (time.perf_counter() - capture_time) * 1000.0
        self.latencies_ms.append(latency_ms)
        logger.info("종단 지연 %.0f ms (mic %d seq %d)", latency_ms, key[0], key[1])

    def _finish(self, connection: socket.socket, receiver: threading.Thread) -> None:
        """보낼 것을 다 보냈으면 송신만 닫고, 서버가 마지막 알림까지 보내고 닫을 때까지 기다린다."""
        self._finishing = True
        # 서버는 남은 오디오를 처리하는 동안에도 PING을 보내므로 수신 타임아웃을 그대로 둔다.
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            return
        receiver.join(client_config.DRAIN_TIMEOUT_SEC)
        if receiver.is_alive():
            logger.warning(
                "서버가 %s초 안에 연결을 닫지 않았습니다",
                client_config.DRAIN_TIMEOUT_SEC,
            )

    def _receive_loop(self, connection: socket.socket) -> None:
        try:
            while True:
                message = read_message(connection)
                self._handle_message(connection, message.message_type, message.body)
        except ConnectionClosed:
            if self._finishing:
                logger.info("서버가 연결을 닫았습니다 (정상 종료)")
            else:
                self._mark_lost("서버가 연결을 닫음")
        except TimeoutError:
            self._mark_lost(f"{self._peer_timeout_sec}초 동안 서버로부터 수신 없음")
        except FramingError as error:
            self._mark_lost(f"메시지 경계 오류: {error}")
        except OSError as error:
            if not self._stop.is_set():
                self._mark_lost(f"수신 오류: {error}")

    def _handle_message(
        self, connection: socket.socket, message_type: MessageType, body: bytes
    ) -> None:
        if message_type == MessageType.ALERT:
            try:
                alert = decode_alert(body)
                self._record_latency(alert)
            except (PayloadError, KeyError, TypeError, ValueError) as error:
                self.ignored_payloads += 1
                logger.warning("ALERT 내용 오류로 무시합니다 — %s", error)
                return
            self.received_alerts.append(alert)
            logger.info(
                "ALERT %s %s: %s", alert["level"], alert["rules"], alert["message"]
            )
            self._outputs.submit_alert(alert)
        elif message_type == MessageType.STATUS:
            try:
                status = decode_status(body)
            except PayloadError as error:
                self.ignored_payloads += 1
                logger.warning("STATUS 내용 오류로 무시합니다 — %s", error)
                return
            self.received_statuses.append(status)
            logger.info(
                "STATUS 빠진 마이크 %s (%d/%d)",
                list(status.missing_mics),
                status.active_mics,
                status.total_mics,
            )
            self._outputs.submit_status(status)
        elif message_type == MessageType.PING:
            try:
                self._send(connection, MessageType.PONG, b"")
            except OSError as error:
                self._mark_lost(f"PONG 송신 실패: {error}")
        elif message_type == MessageType.PONG:
            pass
        else:
            logger.warning("처리하지 않는 메시지 %s 무시", message_type.name)


def parse_mapping(text: str) -> dict[str, str]:
    """ "거실=a.wav,안방=b.wav" → {"거실": "a.wav", "안방": "b.wav"}."""
    mapping = {}
    for item in text.split(","):
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise argparse.ArgumentTypeError(f"'방이름=값' 형식이 아닙니다: {item!r}")
        mapping[name.strip()] = value.strip()
    return mapping


def parse_stall_argument(text: str) -> StallSpec:
    try:
        return parse_stall_spec(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="층간소음 경고 Pi 클라이언트")
    parser.add_argument("--server", default=client_config.SERVER_HOST)
    parser.add_argument("--port", type=int, default=client_config.SERVER_PORT)
    parser.add_argument("--client-id", default=client_config.CLIENT_ID)
    parser.add_argument("--source", choices=["mic", "file"], default="mic")
    parser.add_argument(
        "--files", type=parse_mapping, help="방이름=wav경로 (쉼표 구분)"
    )
    parser.add_argument("--fast", action="store_true", help="파일 모드: 대기 없이 전송")
    parser.add_argument(
        "--stall",
        type=parse_stall_argument,
        help="파일 모드: 방=시작초:길이초 구간 전송 중단 (쉼표로 여러 개, 예: 안방=10:5)",
    )
    parser.add_argument(
        "--no-reconnect", action="store_true", help="끊기면 다시 연결하지 않고 종료"
    )
    parser.add_argument("--log-file", type=Path, help="로그 파일 경로")
    arguments = parser.parse_args(argv)
    if arguments.source == "file" and not arguments.files:
        parser.error("--source file 에는 --files 가 필요합니다")
    if arguments.stall and arguments.source != "file":
        parser.error("--stall 은 파일 모드에서만 쓸 수 있습니다")
    if arguments.source == "mic" and not client_config.MIC_DEVICES:
        parser.error("client_config.MIC_DEVICES가 비어 있습니다")
    return arguments


def build_source(arguments: argparse.Namespace) -> ChunkSource:
    """인자에 맞는 입력을 만든다 (side effect: 파일 읽기)."""
    if arguments.source == "file":
        files = {room: Path(path) for room, path in arguments.files.items()}
        return FileStream(
            files,
            client_config.CHUNK_SAMPLES,
            fast=arguments.fast,
            stalls=arguments.stall,
        )
    return MicStream(
        client_config.MIC_DEVICES,
        client_config.MIC_SAMPLE_RATE,
        client_config.CHUNK_SAMPLES,
    )


def configure_logging(log_file: Path | None) -> None:
    """콘솔(과 파일)로 로그를 남긴다 (side effect: 파일 생성)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    configure_logging(arguments.log_file)
    client = PiClient(
        arguments.server,
        arguments.port,
        arguments.client_id,
        build_source(arguments),
        client_config.CHUNK_SAMPLES,
        reconnect=not arguments.no_reconnect,
    )
    try:
        return client.run()
    except KeyboardInterrupt:
        client.stop()
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
