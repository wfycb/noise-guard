"""TCP 서버: Pi 클라이언트 연결 관리, 수신 스레드, 명령 송신 (side effect: 소켓, 스레드).

연결 하나 = ClientSession 하나. HELLO가 검증을 통과하면 세션을 만들고, 세션의 NetworkSource들을
main.run_sources에 넘긴다. 판단 엔진은 main 쪽에 있으므로 연결이 바뀌어도 상태가 이어진다.

오류 정책 (protocol.py의 예외 구분을 따른다):
- FramingError(경계 깨짐), HELLO에 없는 mic_index, 연결 끊김, 수신 타임아웃 → 연결을 끊는다.
- PayloadError(내용 오류) → 그 메시지만 무시한다. 서버가 받는 JSON은 HELLO뿐이라 거의 해당 없음.
"""

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass

import config
from capture import BacklogOverflowError, NetworkMicStats, NetworkSource, NetworkStream
from protocol import (
    FLAG_OVERFLOW,
    ConnectionClosed,
    FramingError,
    Hello,
    HelloAck,
    MessageType,
    PayloadError,
    decode_audio,
    decode_hello,
    encode_hello_ack,
    read_message,
    send_message,
    validate_hello,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionCounters:
    audio_chunks: int = 0
    pings: int = 0
    ignored_messages: int = 0


class ClientSession:
    """수락된 연결 하나. 수신 스레드가 AUDIO를 NetworkStream에 넣고, 송신 스레드가 큐를 비운다.

    클라이언트가 송신만 닫으면(정상 종료) 입력이 끝난 것으로 보고 스트림만 닫는다. 남은 tick 처리와
    마지막 알림 송신이 끝난 뒤 close()로 소켓을 닫는다.
    """

    def __init__(
        self,
        connection: socket.socket,
        address: tuple[str, int],
        hello: Hello,
        stream_start_time: float,
    ) -> None:
        self.connection = connection
        self.address = address
        self.hello = hello
        self.stream_start_time = stream_start_time
        self.stream = NetworkStream(
            [(mic.index, mic.room, mic.sample_rate) for mic in hello.mics],
            hello.chunk_samples,
            config.CLASSIFY_HOP_SEC,
            config.MIC_STALL_TIMEOUT_SEC,
            config.NETWORK_BACKLOG_MAX_SEC,
        )
        self.counters = SessionCounters()
        self.close_reason: str | None = None
        self._send_queue: queue.Queue[tuple[MessageType, bytes] | None] = queue.Queue()
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name=f"{config.THREAD_NAME_PREFIX}-server-recv",
            daemon=True,
        )
        self._sender = threading.Thread(
            target=self._send_loop,
            name=f"{config.THREAD_NAME_PREFIX}-server-send",
            daemon=True,
        )

    @property
    def sources(self) -> list[NetworkSource]:
        return self.stream.sources

    @property
    def connected(self) -> bool:
        return not self._closed.is_set()

    def stats(self) -> dict[str, NetworkMicStats]:
        return self.stream.stats()

    def start(self) -> None:
        """수신·송신 스레드를 시작한다 (side effect: 스레드)."""
        self._receiver.start()
        self._sender.start()

    def send(self, message_type: MessageType, body: bytes = b"") -> bool:
        """송신 큐에 넣는다. 판단 루프가 소켓 쓰기로 막히지 않게 하기 위해서다."""
        if self._closed.is_set():
            return False
        self._send_queue.put((message_type, body))
        return True

    def _receive_loop(self) -> None:
        reason = "수신 종료"
        try:
            self.connection.settimeout(config.PEER_TIMEOUT_SEC)
            while not self._closed.is_set():
                message = read_message(self.connection)
                self._handle_message(message.message_type, message.body)
        except ConnectionClosed:
            # 클라이언트가 보낼 것을 다 보내고 송신만 닫은 정상 종료일 수 있다.
            # 남은 tick 처리와 알림 송신을 위해 소켓은 열어 두고 입력만 끝낸다.
            logger.info("%s: 클라이언트 입력 종료", self.address)
            self.stream.close()
            return
        except TimeoutError:
            reason = f"{config.PEER_TIMEOUT_SEC}초 동안 수신 없음"
        except (FramingError, BacklogOverflowError) as error:
            reason = f"{type(error).__name__}: {error}"
        except OSError as error:
            reason = f"소켓 오류: {error}"
        if not self._closed.is_set():
            logger.warning("%s: 연결 종료 — %s", self.address, reason)
        self.close(reason, flush=False)

    def _handle_message(self, message_type: MessageType, body: bytes) -> None:
        if message_type == MessageType.AUDIO:
            chunk = decode_audio(body)
            if not self.stream.has_mic(chunk.mic_index):
                raise FramingError(f"HELLO에 없는 mic_index {chunk.mic_index}")
            self.counters.audio_chunks += 1
            self.stream.receive(
                chunk.mic_index,
                chunk.seq,
                chunk.samples,
                overflow=bool(chunk.flags & FLAG_OVERFLOW),
            )
        elif message_type == MessageType.PING:
            self.counters.pings += 1
            self.send(MessageType.PONG)
        elif message_type == MessageType.PONG:
            pass
        elif message_type == MessageType.HELLO:
            raise FramingError("연결 중에 HELLO를 다시 받았습니다")
        else:
            self.counters.ignored_messages += 1
            logger.warning(
                "%s: 처리하지 않는 메시지 %s 무시", self.address, message_type.name
            )

    def _send_loop(self) -> None:
        while True:
            item = self._send_queue.get()
            if item is None:
                return
            message_type, body = item
            try:
                send_message(self.connection, message_type, body)
            except OSError as error:
                logger.warning("%s: 송신 실패 — %s", self.address, error)
                self.close(f"송신 실패: {error}", flush=False)
                return

    def close(self, reason: str | None = None, flush: bool = True) -> None:
        """연결을 닫는다 (side effect: 소켓 종료). flush면 송신 큐를 다 보낸 뒤 닫는다."""
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self.close_reason = reason or "서버가 정상 종료"
        self.stream.close()
        self._send_queue.put(None)
        if flush and threading.current_thread() is not self._sender:
            self._sender.join(config.THREAD_JOIN_TIMEOUT_SEC)
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # 이미 상대가 끊었으면 shutdown이 실패한다.
        self.connection.close()

    def join(self) -> None:
        """스레드가 끝날 때까지 기다린다. close() 이후에 부른다."""
        for thread in (self._receiver, self._sender):
            if thread is not threading.current_thread() and thread.is_alive():
                thread.join(config.THREAD_JOIN_TIMEOUT_SEC)
                if thread.is_alive():
                    logger.warning("%s 스레드가 종료되지 않았습니다", thread.name)


class AudioServer:
    """접속을 받아 HELLO를 검증하고 ClientSession을 넘겨준다. 동시에 클라이언트 하나만 받는다."""

    def __init__(self, host: str, port: int, start_time: float | None = None) -> None:
        self._listener = socket.create_server((host, port))
        self._listener.settimeout(config.SERVER_THREAD_POLL_SEC)
        self.address: tuple[str, int] = self._listener.getsockname()[:2]
        # 가상 시계: --start-time이 있으면 그 시각에서 시작한 것처럼 세션 시작 시각을 옮긴다.
        self._clock_offset = 0.0 if start_time is None else start_time - time.time()
        self._sessions: queue.Queue[ClientSession] = queue.Queue()
        self._active: ClientSession | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name=f"{config.THREAD_NAME_PREFIX}-server-accept",
            daemon=True,
        )

    def start(self) -> None:
        """접속 대기 스레드를 시작한다 (side effect: 스레드)."""
        self._accept_thread.start()
        logger.info("서버 대기 중 %s:%d", *self.address)

    def next_session(self, timeout: float) -> ClientSession | None:
        try:
            return self._sessions.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """접속 대기를 멈추고 활성 세션을 닫는다 (side effect: 소켓·스레드 종료)."""
        self._stop.set()
        self._accept_thread.join(config.THREAD_JOIN_TIMEOUT_SEC)
        self._listener.close()
        with self._lock:
            active = self._active
        if active is not None:
            active.close("서버 종료")
            active.join()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection, address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            self._handshake(connection, address)

    def _handshake(self, connection: socket.socket, address: tuple[str, int]) -> None:
        """HELLO를 받아 검증하고 HELLO_ACK를 보낸다 (side effect: 소켓)."""
        connection.settimeout(config.HELLO_TIMEOUT_SEC)
        try:
            message = read_message(connection)
        except (TimeoutError, ConnectionClosed, FramingError, OSError) as error:
            logger.warning("%s: HELLO 수신 실패 — %s", address, error)
            connection.close()
            return
        # 시간축의 기준은 Pi 시계가 아니라 HELLO를 받은 순간의 노트북 시각이다.
        stream_start_time = time.time() + self._clock_offset
        hello: Hello | None = None
        if message.message_type != MessageType.HELLO:
            reasons = [f"첫 메시지가 HELLO가 아닙니다({message.message_type.name})"]
        else:
            try:
                hello = decode_hello(message.body)
                reasons = validate_hello(hello)
            except PayloadError as error:
                reasons = [str(error)]
        with self._lock:
            if self._active is not None and self._active.connected:
                reasons.append("이미 다른 클라이언트가 연결되어 있습니다")
        accepted = not reasons
        try:
            send_message(
                connection,
                MessageType.HELLO_ACK,
                encode_hello_ack(HelloAck(accepted, tuple(reasons), time.time())),
            )
        except OSError as error:
            logger.warning("%s: HELLO_ACK 송신 실패 — %s", address, error)
            accepted = False
        if not accepted or hello is None:
            logger.warning("%s: 연결 거부 — %s", address, "; ".join(reasons))
            connection.close()
            return
        session = ClientSession(connection, address, hello, stream_start_time)
        with self._lock:
            self._active = session
        session.start()
        logger.info(
            "%s: 연결 수락 client_id=%s 마이크=%s chunk=%d",
            address,
            hello.client_id,
            [(mic.room, mic.sample_rate) for mic in hello.mics],
            hello.chunk_samples,
        )
        self._sessions.put(session)
