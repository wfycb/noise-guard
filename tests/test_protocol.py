import random
import socket
import threading

import numpy
import pytest

from protocol import (
    AUDIO_HEADER,
    FLAG_OVERFLOW,
    LENGTH_PREFIX,
    MAX_CHUNK_SAMPLES,
    MAX_MESSAGE_BYTES,
    MAX_MICS,
    PROTOCOL_VERSION,
    AudioChunk,
    ConnectionClosed,
    FramingError,
    Hello,
    HelloAck,
    Message,
    MessageType,
    MicInfo,
    PayloadError,
    ProtocolError,
    decode_alert,
    decode_audio,
    decode_hello,
    decode_hello_ack,
    encode_alert,
    encode_audio,
    encode_hello,
    encode_hello_ack,
    encode_message,
    read_message,
    recv_exact,
    validate_hello,
)


class FragmentedConnection:
    """recv 한 번에 정해진 크기 이하만 돌려주는 가짜 소켓. TCP의 분할 수신을 흉내 낸다."""

    def __init__(self, data: bytes, fragment_sizes: list[int] | int) -> None:
        self._data = data
        self._position = 0
        self._fragment_sizes = fragment_sizes
        self._call_index = 0

    def recv(self, buffer_size: int) -> bytes:
        if isinstance(self._fragment_sizes, int):
            limit = self._fragment_sizes
        else:
            limit = self._fragment_sizes[self._call_index % len(self._fragment_sizes)]
        self._call_index += 1
        size = min(buffer_size, limit, len(self._data) - self._position)
        chunk = self._data[self._position : self._position + size]
        self._position += size
        return chunk


def make_hello(**overrides: object) -> Hello:
    fields: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "client_id": "pi-1",
        "mics": (MicInfo(0, "거실", 48000), MicInfo(1, "안방", 44100)),
        "chunk_samples": 4800,
    }
    fields.update(overrides)
    return Hello(**fields)  # type: ignore[arg-type]


def make_alert_payload() -> dict[str, object]:
    return {
        "level": "warning",
        "rules": ["R1", "R2"],
        "room": "거실",
        "category": "impact",
        "label": "Walk, footsteps",
        "label_ko": "발소리",
        "peak_db": 61.2,
        "count": 3,
        "room_counts": {"거실": 2, "안방": 1},
        "message": "거실 충격소음 반복 — 최근 1시간 3회",
        "timestamp": 1790000000.0,
        "source_mic_index": 0,
        "source_seq": 123,
    }


def sample_messages() -> list[tuple[MessageType, bytes]]:
    samples = numpy.arange(-2400, 2400, dtype=numpy.int16)
    return [
        (MessageType.HELLO, encode_hello(make_hello())),
        (MessageType.HELLO_ACK, encode_hello_ack(HelloAck(True, (), 1.5))),
        (MessageType.AUDIO, encode_audio(AudioChunk(1, 42, FLAG_OVERFLOW, samples))),
        (MessageType.ALERT, encode_alert(make_alert_payload())),
        (MessageType.PING, b""),
        (MessageType.PONG, b""),
    ]


def read_all(connection: FragmentedConnection, count: int) -> list[Message]:
    return [read_message(connection) for _ in range(count)]


# --- 왕복 ---


def test_hello_round_trip() -> None:
    hello = make_hello()
    assert decode_hello(encode_hello(hello)) == hello


def test_hello_ack_round_trip() -> None:
    ack = HelloAck(False, ("버전 불일치",), 1790000000.25)
    assert decode_hello_ack(encode_hello_ack(ack)) == ack


def test_audio_round_trip_keeps_samples_and_header() -> None:
    samples = numpy.array([-32768, -1, 0, 1, 32767], dtype=numpy.int16)
    decoded = decode_audio(encode_audio(AudioChunk(3, 4_000_000_000, 0, samples)))
    assert (decoded.mic_index, decoded.seq, decoded.flags) == (3, 4_000_000_000, 0)
    numpy.testing.assert_array_equal(decoded.samples, samples)


def test_audio_header_is_little_endian_and_unpadded() -> None:
    body = encode_audio(AudioChunk(1, 2, FLAG_OVERFLOW, numpy.zeros(3, numpy.int16)))
    assert AUDIO_HEADER.size == 11
    assert body[:11] == bytes([1, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0])


def test_length_prefix_is_big_endian_and_includes_type_byte() -> None:
    encoded = encode_message(MessageType.PING, b"abc")
    assert encoded[:4] == (4).to_bytes(4, "big")
    assert encoded[4] == MessageType.PING


def test_alert_round_trip() -> None:
    payload = make_alert_payload()
    assert decode_alert(encode_alert(payload)) == payload


@pytest.mark.parametrize(
    "message_type, body",
    sample_messages(),
    # 기본 ID는 바이트 본문 전체라 Windows 환경변수 길이 제한을 넘는다.
    ids=[message_type.name for message_type, _ in sample_messages()],
)
def test_every_type_survives_framing(message_type: MessageType, body: bytes) -> None:
    message = read_message(
        FragmentedConnection(encode_message(message_type, body), 4096)
    )
    assert message == Message(message_type, body)


# --- 분할·연결 수신 ---


@pytest.mark.parametrize("fragment_size", [1, 7, 4096])
def test_stream_split_into_fixed_fragments(fragment_size: int) -> None:
    messages = sample_messages()
    stream = b"".join(encode_message(t, b) for t, b in messages)
    received = read_all(FragmentedConnection(stream, fragment_size), len(messages))
    assert received == [Message(t, b) for t, b in messages]


@pytest.mark.parametrize("seed", range(5))
def test_stream_split_into_random_fragments(seed: int) -> None:
    rng = random.Random(seed)
    messages = sample_messages() * 3
    stream = b"".join(encode_message(t, b) for t, b in messages)
    sizes = [rng.randint(1, 50) for _ in range(200)]
    received = read_all(FragmentedConnection(stream, sizes), len(messages))
    assert received == [Message(t, b) for t, b in messages]


def test_two_messages_arriving_together_are_split() -> None:
    stream = encode_message(MessageType.PING) + encode_message(MessageType.PONG)
    connection = FragmentedConnection(stream, len(stream))
    assert read_message(connection).message_type == MessageType.PING
    assert read_message(connection).message_type == MessageType.PONG


def test_real_socket_pair_with_sender_thread() -> None:
    server_side, client_side = socket.socketpair()
    messages = sample_messages()
    stream = b"".join(encode_message(t, b) for t, b in messages)

    def send_in_pieces() -> None:
        for start in range(0, len(stream), 13):
            client_side.sendall(stream[start : start + 13])
        client_side.close()

    sender = threading.Thread(target=send_in_pieces)
    sender.start()
    try:
        received = [read_message(server_side) for _ in messages]
        with pytest.raises(ConnectionClosed):
            read_message(server_side)
    finally:
        sender.join()
        server_side.close()
    assert received == [Message(t, b) for t, b in messages]


# --- 오류 ---


def test_length_over_limit_is_rejected_before_reading_body() -> None:
    header = LENGTH_PREFIX.pack(MAX_MESSAGE_BYTES + 1) + bytes([MessageType.PING])
    with pytest.raises(ProtocolError, match="상한"):
        read_message(FragmentedConnection(header, 4096))


def test_encoding_over_limit_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        encode_message(MessageType.ALERT, b"x" * MAX_MESSAGE_BYTES)


def test_zero_length_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        read_message(FragmentedConnection(LENGTH_PREFIX.pack(0), 4096))


def test_unknown_type_is_rejected() -> None:
    stream = LENGTH_PREFIX.pack(1) + bytes([200])
    with pytest.raises(ProtocolError, match="알 수 없는"):
        read_message(FragmentedConnection(stream, 4096))


def test_sample_count_mismatch_is_rejected() -> None:
    body = AUDIO_HEADER.pack(0, 1, 0, 10) + numpy.zeros(9, numpy.int16).tobytes()
    with pytest.raises(ProtocolError, match="sample_count"):
        decode_audio(body)


def test_truncated_audio_header_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        decode_audio(b"\x00\x01")


def test_reserved_flag_bits_are_rejected() -> None:
    body = AUDIO_HEADER.pack(0, 1, 0x2, 0)
    with pytest.raises(ProtocolError, match="flags"):
        decode_audio(body)


def test_audio_longer_than_uint16_cannot_be_encoded() -> None:
    samples = numpy.zeros(MAX_CHUNK_SAMPLES + 1, numpy.int16)
    with pytest.raises(ProtocolError):
        encode_audio(AudioChunk(0, 0, 0, samples))


@pytest.mark.parametrize(
    "body", [b"\xff\xfe", b"[1, 2]", b'{"protocol_version": 1}', b"not json"]
)
def test_malformed_hello_body_is_protocol_error(body: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_hello(body)


def test_alert_missing_field_is_rejected() -> None:
    payload = make_alert_payload()
    del payload["source_seq"]
    with pytest.raises(ProtocolError, match="source_seq"):
        encode_alert(payload)


@pytest.mark.parametrize("cut", [0, 2, 4, 6])
def test_connection_closed_mid_message(cut: int) -> None:
    stream = encode_message(MessageType.ALERT, encode_alert(make_alert_payload()))
    with pytest.raises(ConnectionClosed):
        read_message(FragmentedConnection(stream[:cut], 4096))


def test_recv_exact_reports_partial_progress() -> None:
    with pytest.raises(ConnectionClosed, match="3바이트"):
        recv_exact(FragmentedConnection(b"abc", 1), 5)


# --- HELLO 검증 ---


def test_valid_hello_is_accepted() -> None:
    assert validate_hello(make_hello()) == []


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        ({"protocol_version": 2}, "protocol_version"),
        ({"mics": ()}, "마이크 수"),
        (
            {
                "mics": tuple(
                    MicInfo(index, f"방{index}", 48000) for index in range(MAX_MICS + 1)
                )
            },
            "마이크 수",
        ),
        ({"mics": (MicInfo(0, "거실", 22050),)}, "sample_rate"),
        ({"mics": (MicInfo(0, "거실", 48000), MicInfo(1, "거실", 48000))}, "중복"),
        ({"mics": (MicInfo(0, "  ", 48000),)}, "비어"),
        ({"mics": (MicInfo(0, "거실", 48000), MicInfo(0, "안방", 48000))}, "index"),
        ({"chunk_samples": 100}, "chunk_samples"),
        ({"chunk_samples": 70000}, "chunk_samples"),
        ({"client_id": ""}, "client_id"),
    ],
)
def test_invalid_hello_is_rejected(
    overrides: dict[str, object], expected_reason: str
) -> None:
    reasons = validate_hello(make_hello(**overrides))
    assert reasons and any(expected_reason in reason for reason in reasons)


def test_hello_with_boolean_index_is_protocol_error() -> None:
    body = (
        b'{"protocol_version": 1, "client_id": "pi", "chunk_samples": 4800,'
        b' "mics": [{"index": true, "room": "a", "sample_rate": 48000}]}'
    )
    with pytest.raises(ProtocolError):
        decode_hello(body)


# --- 오류 분류: 경계 오류(연결 끊기) vs 내용 오류(그 메시지만 무시) ---


@pytest.mark.parametrize(
    "stream",
    [
        LENGTH_PREFIX.pack(0),
        LENGTH_PREFIX.pack(MAX_MESSAGE_BYTES + 1),
        LENGTH_PREFIX.pack(1) + bytes([200]),
    ],
    ids=["zero_length", "too_long", "unknown_type"],
)
def test_broken_boundaries_are_framing_errors(stream: bytes) -> None:
    with pytest.raises(FramingError):
        read_message(FragmentedConnection(stream, 4096))


@pytest.mark.parametrize(
    "body",
    [
        b"\x00\x01",
        AUDIO_HEADER.pack(0, 1, 0, 10) + numpy.zeros(9, numpy.int16).tobytes(),
        AUDIO_HEADER.pack(0, 1, 0x2, 0),
    ],
    ids=["short_header", "sample_count_mismatch", "reserved_flags"],
)
def test_broken_audio_is_framing_error(body: bytes) -> None:
    with pytest.raises(FramingError):
        decode_audio(body)


@pytest.mark.parametrize(
    "body",
    [b"{not json", b'{"level": "warning"}', b"[]"],
    ids=["broken_json", "missing_fields", "not_object"],
)
def test_bad_alert_content_is_payload_error(body: bytes) -> None:
    with pytest.raises(PayloadError):
        decode_alert(body)


def test_reader_keeps_going_after_payload_error() -> None:
    # 경계가 정상이면 내용이 깨진 ALERT 다음 메시지도 그대로 읽을 수 있어야 한다.
    stream = encode_message(MessageType.ALERT, b"{broken") + encode_message(
        MessageType.ALERT, encode_alert(make_alert_payload())
    )
    connection = FragmentedConnection(stream, 5)
    with pytest.raises(PayloadError):
        decode_alert(read_message(connection).body)
    assert decode_alert(read_message(connection).body)["room"] == "거실"


def test_framing_and_payload_errors_are_distinct() -> None:
    assert not issubclass(FramingError, PayloadError)
    assert not issubclass(PayloadError, FramingError)
    assert issubclass(FramingError, ProtocolError)
    assert issubclass(PayloadError, ProtocolError)
