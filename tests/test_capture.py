from pathlib import Path

import numpy
from scipy.io import wavfile

from capture import FileSource, RingBuffer, read_wav_mono


def test_read_new_returns_each_sample_once() -> None:
    ring_buffer = RingBuffer(10)
    ring_buffer.write(numpy.arange(4, dtype=numpy.float32))
    first, total = ring_buffer.read_new(0)
    ring_buffer.write(numpy.arange(4, 12, dtype=numpy.float32))
    second, total = ring_buffer.read_new(total)
    numpy.testing.assert_array_equal(
        numpy.concatenate((first, second)), numpy.arange(12)
    )
    empty, _ = ring_buffer.read_new(total)
    assert len(empty) == 0


def test_overrun_keeps_latest_and_counts_all_samples() -> None:
    ring_buffer = RingBuffer(10)
    ring_buffer.write(numpy.arange(25, dtype=numpy.float32))
    samples, total = ring_buffer.read_new(0)
    assert total == 25
    numpy.testing.assert_array_equal(samples, numpy.arange(15, 25))


def test_latest_returns_none_until_enough_samples() -> None:
    ring_buffer = RingBuffer(10)
    ring_buffer.write(numpy.arange(3, dtype=numpy.float32))
    assert ring_buffer.latest(5) is None
    ring_buffer.write(numpy.arange(3, 13, dtype=numpy.float32))
    numpy.testing.assert_array_equal(ring_buffer.latest(5), numpy.arange(8, 13))


def write_test_wav(path: Path, samples: numpy.ndarray, sample_rate: int) -> None:
    wavfile.write(path, sample_rate, samples.astype(numpy.float32))


def test_file_source_keeps_sample_rate_and_streams_in_order(tmp_path: Path) -> None:
    path = tmp_path / "ramp.wav"
    write_test_wav(path, numpy.arange(100) / 100.0, 10)
    source = FileSource("거실", path, end_behavior="stop")
    assert source.sample_rate == 10
    source.advance(3.0)
    first, total = source.read_new(0)
    source.advance(3.0)
    second, total = source.read_new(total)
    numpy.testing.assert_allclose(
        numpy.concatenate((first, second)), numpy.arange(60) / 100.0, atol=1e-6
    )


def test_file_source_pad_fills_silence_after_end(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    write_test_wav(path, numpy.ones(15), 10)
    source = FileSource("거실", path, end_behavior="pad")
    source.advance(2.0)
    assert source.exhausted
    samples, _ = source.read_new(0)
    numpy.testing.assert_array_equal(samples, [1.0] * 15 + [0.0] * 5)


def test_file_source_stop_adds_nothing_after_end(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    write_test_wav(path, numpy.ones(15), 10)
    source = FileSource("거실", path, end_behavior="stop")
    source.advance(2.0)
    _, total = source.read_new(0)
    source.advance(1.0)
    samples, _ = source.read_new(total)
    assert len(samples) == 0


def test_int16_wav_is_scaled_to_unit_range(tmp_path: Path) -> None:
    path = tmp_path / "int16.wav"
    wavfile.write(path, 10, numpy.array([-32768, 0, 16384], dtype=numpy.int16))
    samples, _ = read_wav_mono(path)
    numpy.testing.assert_allclose(samples, [-1.0, 0.0, 0.5])
