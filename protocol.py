"""서버(노트북)·클라이언트(Pi) 공용 메시지 규격. numpy와 표준 라이브러리만 쓴다.

pi_client가 import하므로 config.py 등 서버 모듈을 import하지 않는다. 그래서 규격 상수는
config.py가 아니라 여기에 둔다 (양쪽이 반드시 같은 값을 써야 하는 값들이다).

프레이밍:
    [4바이트 big-endian uint32: 본문 길이(타입 1바이트 포함)][1바이트 메시지 타입][본문]

엔디안이 섞여 있는 것은 의도다. 길이 접두는 네트워크 바이트 순서(big-endian) 관례를 따르고,
AUDIO 헤더와 PCM은 x86·ARM(Pi) 모두의 기본인 little-endian으로 두어 numpy 변환 비용을 없앤다.
"""

import json
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol

import numpy

PROTOCOL_VERSION = 1
# 잘못된 길이 값으로 메모리가 폭주하지 않게 막는 상한. 48kHz 100ms 청크(약 9.6KB)보다 충분히 크다.
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_MICS = 8
ALLOWED_SAMPLE_RATES: tuple[int, ...] = (16000, 44100, 48000)
# 하한: 청크가 너무 작으면 메시지 수가 폭증한다(16kHz 기준 10ms). 상한: AUDIO 헤더 sample_count가 uint16.
MIN_CHUNK_SAMPLES = 160
MAX_CHUNK_SAMPLES = 65535
MAX_MIC_INDEX = 255  # AUDIO 헤더 mic_index가 uint8

LENGTH_PREFIX = struct.Struct(">I")
TYPE_BYTE = struct.Struct("B")
# (mic_index uint8, seq uint32, flags uint32, sample_count uint16), little-endian, 패딩 없음.
AUDIO_HEADER = struct.Struct("<BIIH")
PCM_DTYPE = numpy.dtype("<i2")

FLAG_OVERFLOW = 0x00000001  # 이 청크 구간에서 Pi 오디오 콜백 overflow 발생
KNOWN_FLAGS = FLAG_OVERFLOW


class MessageType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    AUDIO = 3
    ALERT = 4
    PING = 5
    PONG = 6
    STATUS = 7


class ProtocolError(Exception):
    """규격 위반의 공통 부모. 받는 쪽은 아래 두 하위 클래스로 처리 정책을 나눈다."""


class FramingError(ProtocolError):
    """메시지 경계가 깨졌다(길이, 타입, AUDIO 헤더·sample_count, 예약 flags).

    다음 메시지의 시작 위치를 더 이상 믿을 수 없으므로 연결을 끊는다.
    """


class PayloadError(ProtocolError):
    """경계는 정상인데 본문 내용이 잘못됐다(JSON 깨짐, 필드 누락·형식 오류).

    다음 메시지는 정상적으로 읽을 수 있으므로, ALERT 같은 메시지는 경고 후 무시하고 계속한다.
    HELLO는 핸드셰이크라 내용이 잘못되면 거부한다.
    """


class ConnectionClosed(Exception):
    """상대가 메시지 도중이나 메시지 사이에서 연결을 닫았다."""


class Receivable(Protocol):
    def recv(self, buffer_size: int) -> bytes: ...


@dataclass(frozen=True)
class Message:
    message_type: MessageType
    body: bytes


@dataclass(frozen=True)
class MicInfo:
    index: int
    room: str
    sample_rate: int


@dataclass(frozen=True)
class Hello:
    protocol_version: int
    client_id: str
    mics: tuple[MicInfo, ...]
    chunk_samples: int


@dataclass(frozen=True)
class HelloAck:
    accepted: bool
    reasons: tuple[str, ...]
    server_time: float


@dataclass(frozen=True)
class AudioChunk:
    mic_index: int
    seq: int
    flags: int
    samples: numpy.ndarray  # int16, 모노


# --- 프레이밍 ---


def encode_message(message_type: MessageType, body: bytes = b"") -> bytes:
    """길이 접두 + 타입 + 본문. 상한을 넘으면 보내기 전에 ProtocolError."""
    length = TYPE_BYTE.size + len(body)
    if length > MAX_MESSAGE_BYTES:
        raise FramingError(
            f"메시지 길이 {length}가 상한 {MAX_MESSAGE_BYTES}를 넘습니다"
        )
    return LENGTH_PREFIX.pack(length) + TYPE_BYTE.pack(message_type) + body


def recv_exact(connection: Receivable, byte_count: int) -> bytes:
    """정확히 byte_count 바이트를 받을 때까지 읽는다 (side effect: 소켓 읽기).

    TCP recv는 요청보다 적게 돌려줄 수 있으므로 반복해서 모은다. 0바이트면 상대가 닫은 것이다.
    """
    chunks = []
    remaining = byte_count
    while remaining > 0:
        received = connection.recv(remaining)
        if not received:
            raise ConnectionClosed(
                f"{byte_count}바이트 중 {byte_count - remaining}바이트 받은 상태에서 연결 종료"
            )
        chunks.append(received)
        remaining -= len(received)
    return b"".join(chunks)


def read_message(connection: Receivable) -> Message:
    """메시지 하나를 읽는다 (side effect: 소켓 읽기). 규격 위반은 ProtocolError."""
    (length,) = LENGTH_PREFIX.unpack(recv_exact(connection, LENGTH_PREFIX.size))
    if length < TYPE_BYTE.size:
        raise FramingError(f"메시지 길이 {length}가 타입 바이트보다 짧습니다")
    if length > MAX_MESSAGE_BYTES:
        raise FramingError(
            f"메시지 길이 {length}가 상한 {MAX_MESSAGE_BYTES}를 넘습니다"
        )
    payload = recv_exact(connection, length)
    (type_value,) = TYPE_BYTE.unpack(payload[: TYPE_BYTE.size])
    try:
        message_type = MessageType(type_value)
    except ValueError as error:
        raise FramingError(f"알 수 없는 메시지 타입 {type_value}") from error
    return Message(message_type, payload[TYPE_BYTE.size :])


def send_message(
    connection: socket.socket, message_type: MessageType, body: bytes = b""
) -> None:
    """메시지 하나를 보낸다 (side effect: 소켓 쓰기)."""
    connection.sendall(encode_message(message_type, body))


# --- JSON 본문 ---


def encode_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def decode_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PayloadError(f"JSON 본문을 해석할 수 없습니다: {error}") from error
    if not isinstance(payload, dict):
        raise PayloadError("JSON 본문은 객체여야 합니다")
    return payload


# --- HELLO / HELLO_ACK ---


def encode_hello(hello: Hello) -> bytes:
    return encode_json(
        {
            "protocol_version": hello.protocol_version,
            "client_id": hello.client_id,
            "mics": [
                {"index": mic.index, "room": mic.room, "sample_rate": mic.sample_rate}
                for mic in hello.mics
            ],
            "chunk_samples": hello.chunk_samples,
        }
    )


def decode_hello(body: bytes) -> Hello:
    """HELLO 본문을 읽는다. 필드 타입이 틀리면 ProtocolError (값 검증은 validate_hello)."""
    payload = decode_json(body)
    try:
        mics = tuple(
            MicInfo(
                index=_require_int(mic["index"]),
                room=_require_str(mic["room"]),
                sample_rate=_require_int(mic["sample_rate"]),
            )
            for mic in payload["mics"]
        )
        return Hello(
            protocol_version=_require_int(payload["protocol_version"]),
            client_id=_require_str(payload["client_id"]),
            mics=mics,
            chunk_samples=_require_int(payload["chunk_samples"]),
        )
    except (KeyError, TypeError) as error:
        raise PayloadError(f"HELLO 필드가 없거나 형식이 틀립니다: {error}") from error


def _require_int(value: Any) -> int:
    # bool은 int의 하위 타입이라 따로 막는다.
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"정수가 아닙니다: {value!r}")
    return value


def _require_str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"문자열이 아닙니다: {value!r}")
    return value


def validate_hello(hello: Hello) -> list[str]:
    """거부 사유 목록. 비어 있으면 수락 (순수 함수)."""
    reasons = []
    if hello.protocol_version != PROTOCOL_VERSION:
        reasons.append(
            f"protocol_version {hello.protocol_version} ≠ {PROTOCOL_VERSION}"
        )
    if not hello.client_id.strip():
        reasons.append("client_id가 비어 있습니다")
    if not 1 <= len(hello.mics) <= MAX_MICS:
        reasons.append(f"마이크 수 {len(hello.mics)}가 1~{MAX_MICS} 범위 밖입니다")
    if not MIN_CHUNK_SAMPLES <= hello.chunk_samples <= MAX_CHUNK_SAMPLES:
        reasons.append(
            f"chunk_samples {hello.chunk_samples}가 "
            f"{MIN_CHUNK_SAMPLES}~{MAX_CHUNK_SAMPLES} 범위 밖입니다"
        )
    rooms = [mic.room.strip() for mic in hello.mics]
    if any(not room for room in rooms):
        reasons.append("방 이름이 비어 있는 마이크가 있습니다")
    if len(set(rooms)) != len(rooms):
        reasons.append("방 이름이 중복됩니다")
    indices = [mic.index for mic in hello.mics]
    if len(set(indices)) != len(indices):
        reasons.append("마이크 index가 중복됩니다")
    for mic in hello.mics:
        if not 0 <= mic.index <= MAX_MIC_INDEX:
            reasons.append(
                f"마이크 index {mic.index}가 0~{MAX_MIC_INDEX} 범위 밖입니다"
            )
        if mic.sample_rate not in ALLOWED_SAMPLE_RATES:
            reasons.append(
                f"{mic.room}: sample_rate {mic.sample_rate}는 허용 목록 "
                f"{ALLOWED_SAMPLE_RATES}에 없습니다"
            )
    return reasons


def encode_hello_ack(ack: HelloAck) -> bytes:
    return encode_json(
        {
            "accepted": ack.accepted,
            "reasons": list(ack.reasons),
            "server_time": ack.server_time,
        }
    )


def decode_hello_ack(body: bytes) -> HelloAck:
    payload = decode_json(body)
    try:
        accepted = payload["accepted"]
        if not isinstance(accepted, bool):
            raise TypeError(f"accepted가 bool이 아닙니다: {accepted!r}")
        return HelloAck(
            accepted=accepted,
            reasons=tuple(_require_str(reason) for reason in payload["reasons"]),
            server_time=float(payload["server_time"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PayloadError(f"HELLO_ACK 형식이 틀립니다: {error}") from error


# --- AUDIO ---


def encode_audio(chunk: AudioChunk) -> bytes:
    """AUDIO 본문 = 고정 헤더 + int16 little-endian PCM."""
    samples = numpy.ascontiguousarray(chunk.samples, dtype=PCM_DTYPE)
    if samples.ndim != 1 or len(samples) > MAX_CHUNK_SAMPLES:
        raise FramingError(f"PCM은 1차원, 최대 {MAX_CHUNK_SAMPLES}샘플이어야 합니다")
    header = AUDIO_HEADER.pack(chunk.mic_index, chunk.seq, chunk.flags, len(samples))
    return header + samples.tobytes()


def decode_audio(body: bytes) -> AudioChunk:
    """AUDIO 본문을 읽는다. 헤더의 sample_count와 실제 길이가 다르면 ProtocolError."""
    if len(body) < AUDIO_HEADER.size:
        raise FramingError(f"AUDIO 본문 {len(body)}바이트가 헤더보다 짧습니다")
    mic_index, seq, flags, sample_count = AUDIO_HEADER.unpack_from(body)
    pcm_bytes = body[AUDIO_HEADER.size :]
    if len(pcm_bytes) != sample_count * PCM_DTYPE.itemsize:
        raise FramingError(
            f"sample_count {sample_count}와 PCM 길이 {len(pcm_bytes)}바이트가 맞지 않습니다"
        )
    if flags & ~KNOWN_FLAGS:
        raise FramingError(f"예약된 flags 비트가 설정됨: {flags:#x}")
    samples = numpy.frombuffer(pcm_bytes, dtype=PCM_DTYPE)
    return AudioChunk(mic_index=mic_index, seq=seq, flags=flags, samples=samples)


# --- ALERT ---

ALERT_REQUIRED_FIELDS: tuple[str, ...] = (
    "level",
    "rules",
    "room",
    "category",
    "label",
    "label_ko",
    "peak_db",
    "count",
    "room_counts",
    "message",
    "timestamp",
    "source_mic_index",
    "source_seq",
)


def encode_alert(payload: dict[str, Any]) -> bytes:
    missing = [field for field in ALERT_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise PayloadError(f"ALERT 필수 필드 누락: {missing}")
    return encode_json(payload)


def decode_alert(body: bytes) -> dict[str, Any]:
    payload = decode_json(body)
    missing = [field for field in ALERT_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise PayloadError(f"ALERT 필수 필드 누락: {missing}")
    return payload


# --- STATUS (노트북 → Pi, 마이크 상태가 바뀔 때만) ---


@dataclass(frozen=True)
class StatusMessage:
    missing_mics: tuple[str, ...]  # MIC_MISSING_WARN_TICKS 연속으로 tick에서 빠진 방
    active_mics: int  # 이번 tick에 레벨·분류를 낸 마이크 수
    total_mics: int
    timestamp: float


def encode_status(status: StatusMessage) -> bytes:
    return encode_json(
        {
            "missing_mics": list(status.missing_mics),
            "active_mics": status.active_mics,
            "total_mics": status.total_mics,
            "timestamp": status.timestamp,
        }
    )


def decode_status(body: bytes) -> StatusMessage:
    payload = decode_json(body)
    try:
        missing = payload["missing_mics"]
        if not isinstance(missing, list):
            raise TypeError(f"missing_mics가 목록이 아닙니다: {missing!r}")
        return StatusMessage(
            missing_mics=tuple(_require_str(room) for room in missing),
            active_mics=_require_int(payload["active_mics"]),
            total_mics=_require_int(payload["total_mics"]),
            timestamp=float(payload["timestamp"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PayloadError(f"STATUS 형식이 틀립니다: {error}") from error
