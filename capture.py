"""오디오 입력 소스(마이크/파일)와 소스별 링버퍼 (side effect: 오디오 장치·파일 접근).

MicSource와 FileSource는 같은 AudioSource 인터페이스를 가져 main.py에서 교체할 수 있다.
"""

import threading
from pathlib import Path
from typing import Literal, Protocol

import numpy
import sounddevice
from scipy.io import wavfile

import config


class RingBuffer:
    """오디오 콜백 스레드가 쓰고 워커 스레드가 읽는 고정 크기 원형 버퍼."""

    def __init__(self, capacity_samples: int) -> None:
        self._samples = numpy.zeros(capacity_samples, dtype=numpy.float32)
        self._write_position = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def write(self, samples: numpy.ndarray) -> None:
        """샘플을 덧붙인다. 용량을 넘으면 가장 오래된 샘플을 덮어쓴다."""
        capacity = len(self._samples)
        incoming_count = len(samples)
        if incoming_count >= capacity:
            samples = samples[-capacity:]
        with self._lock:
            end_position = self._write_position + len(samples)
            if end_position <= capacity:
                self._samples[self._write_position : end_position] = samples
            else:
                first_part_length = capacity - self._write_position
                self._samples[self._write_position :] = samples[:first_part_length]
                self._samples[: end_position - capacity] = samples[first_part_length:]
            self._write_position = end_position % capacity
            self._total_written += incoming_count

    def latest(self, num_samples: int) -> numpy.ndarray | None:
        """가장 최근 num_samples개를 시간순 복사본으로 반환한다. 아직 모자라면 None."""
        capacity = len(self._samples)
        if num_samples > capacity:
            raise ValueError(
                f"요청 길이 {num_samples}가 버퍼 용량 {capacity}보다 큽니다"
            )
        with self._lock:
            if self._total_written < num_samples:
                return None
            return self._copy_latest_locked(num_samples)

    def read_new(self, last_total_written: int) -> tuple[numpy.ndarray, int]:
        """last_total_written 이후 새로 들어온 샘플과 현재 total_written을 함께 반환한다.

        레벨 필터 상태를 이으려면 각 샘플을 정확히 한 번씩 읽어야 하므로, 길이 계산과
        복사를 한 잠금 안에서 한다. 용량보다 많이 밀렸으면 남아 있는 만큼만 돌려준다.
        """
        with self._lock:
            new_count = min(
                self._total_written - last_total_written, len(self._samples)
            )
            return self._copy_latest_locked(new_count), self._total_written

    def _copy_latest_locked(self, num_samples: int) -> numpy.ndarray:
        capacity = len(self._samples)
        start_position = (self._write_position - num_samples) % capacity
        if start_position + num_samples <= capacity:
            return self._samples[start_position : start_position + num_samples].copy()
        return numpy.concatenate(
            (self._samples[start_position:], self._samples[: self._write_position])
        )


FileEndBehavior = Literal["pad", "stop"]


class AudioSource(Protocol):
    """main.py가 쓰는 입력 소스 인터페이스."""

    name: str
    sample_rate: int

    @property
    def overflow_count(self) -> int: ...

    @property
    def exhausted(self) -> bool: ...

    def start(self) -> None: ...

    def advance(self, duration_sec: float) -> None: ...

    def read_new(self, last_total_written: int) -> tuple[numpy.ndarray, int]: ...

    def latest(self, num_samples: int) -> numpy.ndarray | None: ...

    def close(self) -> None: ...


class MicSource:
    """장치 하나의 InputStream을 열어 링버퍼에 채운다 (side effect: 오디오 장치 접근).

    콜백에서는 버퍼에 복사만 한다. 무거운 연산을 하면 overflow가 난다.
    """

    def __init__(self, name: str, device: int | str) -> None:
        self.name = name
        self.device = device
        self.sample_rate = config.CAPTURE_SAMPLE_RATE
        self._ring_buffer = RingBuffer(int(config.RING_BUFFER_SEC * self.sample_rate))
        self._overflow_count = 0
        self._stream = sounddevice.InputStream(
            device=device,
            channels=config.CAPTURE_CHANNELS,
            samplerate=self.sample_rate,
            dtype=config.CAPTURE_DTYPE,
            callback=self._on_audio_block,
        )

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def exhausted(self) -> bool:
        return False

    def _on_audio_block(
        self,
        input_data: numpy.ndarray,
        frame_count: int,
        time_info: object,
        status: sounddevice.CallbackFlags,
    ) -> None:
        if status.input_overflow:
            self._overflow_count += 1
        self._ring_buffer.write(input_data[:, 0])

    def start(self) -> None:
        """스트림 수집을 시작한다 (side effect: 장치 시작)."""
        self._stream.start()

    def advance(self, duration_sec: float) -> None:
        """마이크는 콜백이 실시간으로 채우므로 할 일이 없다."""

    def read_new(self, last_total_written: int) -> tuple[numpy.ndarray, int]:
        return self._ring_buffer.read_new(last_total_written)

    def latest(self, num_samples: int) -> numpy.ndarray | None:
        return self._ring_buffer.latest(num_samples)

    def close(self) -> None:
        """스트림을 멈추고 장치를 해제한다 (side effect: 장치 해제)."""
        self._stream.stop()
        self._stream.close()


def read_wav_mono(path: Path) -> tuple[numpy.ndarray, int]:
    """wav를 [-1, 1] float32 모노로 읽는다 (side effect: 파일 읽기)."""
    sample_rate, samples = wavfile.read(path)
    if numpy.issubdtype(samples.dtype, numpy.integer):
        full_scale = float(numpy.iinfo(samples.dtype).max) + 1.0
        if samples.dtype == numpy.uint8:
            # 8bit wav는 부호 없는 정수라 중앙값(128)을 0으로 옮겨야 한다.
            samples = samples.astype(numpy.float32) - full_scale / 2.0
            full_scale /= 2.0
        samples = samples.astype(numpy.float32) / full_scale
    samples = samples.astype(numpy.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples, int(sample_rate)


class FileSource:
    """wav 파일 하나를 마이크 하나처럼 공급한다 (side effect: 생성 시 파일 읽기).

    advance(duration)가 호출될 때마다 그만큼의 샘플을 링버퍼에 넣는다. 재생 속도(실시간/고속)는
    호출하는 쪽이 정하므로, 이 클래스는 시계를 모른다. 샘플레이트는 파일 원래 값을 유지한다.
    """

    def __init__(self, name: str, path: Path, end_behavior: FileEndBehavior) -> None:
        self.name = name
        self.path = path
        self._samples, self.sample_rate = read_wav_mono(path)
        self._end_behavior = end_behavior
        self._position = 0
        self._ring_buffer = RingBuffer(int(config.RING_BUFFER_SEC * self.sample_rate))

    @property
    def overflow_count(self) -> int:
        return 0

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self._samples)

    def start(self) -> None:
        """파일은 이미 메모리에 있으므로 할 일이 없다."""

    def advance(self, duration_sec: float) -> None:
        """다음 duration_sec 만큼의 샘플을 버퍼에 넣는다. 끝났으면 pad는 무음, stop은 아무것도 안 넣음."""
        sample_count = round(duration_sec * self.sample_rate)
        chunk = self._samples[self._position : self._position + sample_count]
        self._position += len(chunk)
        if len(chunk) < sample_count and self._end_behavior == "pad":
            chunk = numpy.concatenate(
                (chunk, numpy.zeros(sample_count - len(chunk), dtype=numpy.float32))
            )
        if len(chunk):
            self._ring_buffer.write(chunk)

    def read_new(self, last_total_written: int) -> tuple[numpy.ndarray, int]:
        return self._ring_buffer.read_new(last_total_written)

    def latest(self, num_samples: int) -> numpy.ndarray | None:
        return self._ring_buffer.latest(num_samples)

    def close(self) -> None:
        """파일 소스는 해제할 자원이 없다."""
