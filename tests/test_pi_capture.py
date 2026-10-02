"""pi_client.capture의 wav 파서와 송신 큐 테스트. 기대값은 scipy로 만든다(테스트에서만 사용)."""

import struct
import threading
from pathlib import Path

import numpy
import pytest
from scipy.io import wavfile

from pi_client.capture import (
    KSDATAFORMAT_GUID_TAIL,
    WAVE_FORMAT_EXTENSIBLE,
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    AudioChunkItem,
    ChunkQueue,
    FileStream,
    UnsupportedWavError,
    parse_stall_spec,
    parse_wav,
    read_wav_int16,
)

SAMPLE_RATE = 48000


def expected_int16_from_float(samples: numpy.ndarray) -> numpy.ndarray:
    scaled = numpy.round(numpy.clip(samples.astype(numpy.float64), -1.0, 1.0) * 32768.0)
    return numpy.clip(scaled, -32768, 32767).astype(numpy.int16)


def build_wav(
    format_tag: int,
    bits: int,
    pcm: bytes,
    channels: int = 1,
    extensible_subformat: int | None = None,
) -> bytes:
    """fmt/data 청크를 직접 조립한다. EXTENSIBLE은 scipy가 쓰지 않으므로 이렇게 만든다."""
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        SAMPLE_RATE,
        SAMPLE_RATE * block_align,
        block_align,
        bits,
    )
    if extensible_subformat is not None:
        fmt += struct.pack(
            "<HHIH14s", 22, bits, 0x4, extensible_subformat, KSDATAFORMAT_GUID_TAIL
        )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_pcm16_matches_scipy(tmp_path: Path) -> None:
    samples = (
        numpy.random.default_rng(0).integers(-32768, 32767, 1000).astype(numpy.int16)
    )
    path = tmp_path / "pcm16.wav"
    wavfile.write(path, SAMPLE_RATE, samples)
    _, expected = wavfile.read(path)
    audio = read_wav_int16(path)
    assert audio.sample_rate == SAMPLE_RATE
    assert audio.clipped_samples == 0
    numpy.testing.assert_array_equal(audio.samples, expected)


def test_float32_matches_scipy(tmp_path: Path) -> None:
    samples = numpy.linspace(-0.99, 0.99, 1001).astype(numpy.float32)
    path = tmp_path / "float32.wav"
    wavfile.write(path, SAMPLE_RATE, samples)
    _, expected = wavfile.read(path)
    audio = read_wav_int16(path)
    assert audio.clipped_samples == 0
    numpy.testing.assert_array_equal(audio.samples, expected_int16_from_float(expected))


@pytest.mark.parametrize(
    "subformat, bits, dtype",
    [(WAVE_FORMAT_PCM, 16, "<i2"), (WAVE_FORMAT_IEEE_FLOAT, 32, "<f4")],
    ids=["pcm16", "float32"],
)
def test_extensible_matches_scipy(
    tmp_path: Path, subformat: int, bits: int, dtype: str
) -> None:
    if dtype == "<i2":
        samples = numpy.arange(-500, 500, dtype=dtype)
    else:
        samples = numpy.linspace(-0.5, 0.5, 1000).astype(dtype)
    path = tmp_path / "extensible.wav"
    path.write_bytes(
        build_wav(
            WAVE_FORMAT_EXTENSIBLE,
            bits,
            samples.tobytes(),
            extensible_subformat=subformat,
        )
    )
    _, expected = wavfile.read(path)
    audio = read_wav_int16(path)
    if dtype == "<i2":
        numpy.testing.assert_array_equal(audio.samples, expected)
    else:
        numpy.testing.assert_array_equal(
            audio.samples, expected_int16_from_float(expected)
        )


def test_float_clipping_is_counted() -> None:
    samples = numpy.array([0.0, 1.5, -2.0, 0.5, 1.0, -1.0], dtype="<f4")
    audio = parse_wav(build_wav(WAVE_FORMAT_IEEE_FLOAT, 32, samples.tobytes()))
    assert audio.clipped_samples == 2
    numpy.testing.assert_array_equal(
        audio.samples, [0, 32767, -32768, 16384, 32767, -32768]
    )


def test_stereo_is_averaged_to_mono() -> None:
    frames = numpy.array([[100, 300], [-200, -400]], dtype="<i2")
    audio = parse_wav(build_wav(WAVE_FORMAT_PCM, 16, frames.tobytes(), channels=2))
    numpy.testing.assert_array_equal(audio.samples, [200, -300])


def test_odd_sized_chunk_padding_is_skipped() -> None:
    # 홀수 크기 청크 뒤의 패딩 바이트를 건너뛰어야 fmt/data를 찾을 수 있다.
    pcm = numpy.array([1, 2, 3], dtype="<i2").tobytes()
    wav = build_wav(WAVE_FORMAT_PCM, 16, pcm)
    riff_body = wav[12:]
    odd_chunk = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\x00"
    data = b"RIFF" + struct.pack("<I", 4 + len(odd_chunk) + len(riff_body)) + b"WAVE"
    audio = parse_wav(data + odd_chunk + riff_body)
    numpy.testing.assert_array_equal(audio.samples, [1, 2, 3])


@pytest.mark.parametrize(
    "data, message",
    [
        (b"not a wav", "RIFF"),
        (build_wav(WAVE_FORMAT_PCM, 24, b"\x00" * 6), "지원하지 않는"),
        (build_wav(WAVE_FORMAT_IEEE_FLOAT, 64, b"\x00" * 16), "지원하지 않는"),
        (build_wav(0x0006, 8, b"\x00" * 4), "지원하지 않는"),  # A-law
        (
            build_wav(
                WAVE_FORMAT_EXTENSIBLE, 16, b"\x00" * 4, extensible_subformat=0x0006
            ),
            "지원하지 않는",
        ),
    ],
    ids=["not_riff", "pcm24", "float64", "alaw", "extensible_alaw"],
)
def test_unsupported_formats_raise_clear_error(data: bytes, message: str) -> None:
    with pytest.raises(UnsupportedWavError, match=message):
        parse_wav(data)


def test_extensible_with_unknown_guid_is_rejected() -> None:
    wav = bytearray(
        build_wav(WAVE_FORMAT_EXTENSIBLE, 16, b"\x00" * 4, extensible_subformat=1)
    )
    guid_offset = wav.index(KSDATAFORMAT_GUID_TAIL)
    wav[guid_offset] ^= 0xFF
    with pytest.raises(UnsupportedWavError, match="GUID"):
        parse_wav(bytes(wav))


def make_item(seq: int) -> AudioChunkItem:
    return AudioChunkItem(0, seq, 0, numpy.zeros(4, numpy.int16), 0.0)


def test_queue_drops_oldest_when_full() -> None:
    chunk_queue = ChunkQueue(max_chunks=3)
    for seq in range(5):
        chunk_queue.put_drop_oldest(make_item(seq))
    assert chunk_queue.dropped_chunks == 2
    assert [chunk_queue.get(0.0).seq for _ in range(3)] == [2, 3, 4]


def test_blocking_put_waits_for_space() -> None:
    chunk_queue = ChunkQueue(max_chunks=1)
    stop_event = threading.Event()
    chunk_queue.put_blocking(make_item(0), stop_event)
    putter = threading.Thread(
        target=chunk_queue.put_blocking, args=(make_item(1), stop_event)
    )
    putter.start()
    assert chunk_queue.get(1.0).seq == 0
    putter.join(2.0)
    assert not putter.is_alive()
    assert chunk_queue.get(1.0).seq == 1
    assert chunk_queue.dropped_chunks == 0


# --- FileStream --stall, seq 재설정 ---


def test_parse_stall_spec_supports_multiple_entries() -> None:
    assert parse_stall_spec("안방=10:5,서재=20:3,안방=40:2") == {
        "안방": [(10.0, 5.0), (40.0, 2.0)],
        "서재": [(20.0, 3.0)],
    }


@pytest.mark.parametrize("text", ["안방", "안방=10", "안방=a:5", "안방=10:0", "=1:2"])
def test_parse_stall_spec_rejects_bad_text(text: str) -> None:
    with pytest.raises(ValueError):
        parse_stall_spec(text)


def drain(chunk_queue: ChunkQueue) -> list[AudioChunkItem]:
    items = []
    while (item := chunk_queue.get(0.0)) is not None:
        items.append(item)
    return items


def test_stall_drops_chunks_but_seq_keeps_counting(tmp_path: Path) -> None:
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, numpy.ones(SAMPLE_RATE * 2, numpy.int16))
    stream = FileStream({"안방": path}, 4800, fast=True, stalls={"안방": [(0.5, 0.5)]})
    chunk_queue = ChunkQueue(max_chunks=100)
    stream.start(chunk_queue, threading.Event())
    stream.stop()
    seqs = [item.seq for item in drain(chunk_queue)]
    # 100ms 청크 20개 중 0.5~1.0초 구간(seq 5~9)만 빠진다.
    assert seqs == [0, 1, 2, 3, 4] + list(range(10, 20))
    assert stream.stalled_chunks == 5


def test_stall_for_unknown_room_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, numpy.zeros(4800, numpy.int16))
    with pytest.raises(ValueError, match="서재"):
        FileStream({"안방": path}, 4800, fast=True, stalls={"서재": [(1.0, 1.0)]})


def test_reset_sequence_restarts_seq_but_not_file_position(tmp_path: Path) -> None:
    path = tmp_path / "room.wav"
    samples = numpy.arange(4800 * 6, dtype=numpy.int16)
    wavfile.write(path, SAMPLE_RATE, samples)
    # 실시간 재생(100ms마다 1청크)이라 재설정 시점 이후의 청크는 아직 만들어지지 않았다.
    stream = FileStream({"안방": path}, 4800, fast=False)
    chunk_queue = ChunkQueue(max_chunks=10)
    stream.start(chunk_queue, threading.Event())
    first = [chunk_queue.get(1.0), chunk_queue.get(1.0)]
    stream.reset_sequence()
    stream.stop()
    rest = drain(chunk_queue)
    assert [item.seq for item in first] == [0, 1]
    assert [item.seq for item in rest] == [0, 1, 2, 3]
    starts = [int(item.samples[0]) for item in first + rest]
    assert starts == [0, 4800, 9600, 14400, 19200, 24000]
