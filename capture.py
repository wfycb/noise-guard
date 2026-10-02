"""오디오 입력 소스(마이크/파일)와 소스별 링버퍼 (side effect: 오디오 장치·파일 접근).

MicSource, FileSource, NetworkSource는 같은 AudioSource 인터페이스를 가져 main.py에서
교체할 수 있다.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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


# --- 네트워크 소스 (Pi가 보낸 PCM) ---

PCM_FULL_SCALE = 32768.0  # int16 → [-1, 1) 변환. read_wav_mono와 같은 기준이다.


class BacklogOverflowError(Exception):
    """처리하지 못한 수신 오디오가 상한을 넘었다. 서버가 실시간을 따라가지 못하는 상태다."""


class InboundBuffer:
    """도착했지만 아직 소비하지 않은 오디오. 절대 샘플 위치 [start, end)를 담는다."""

    def __init__(self) -> None:
        self._chunks: deque[numpy.ndarray] = deque()
        self.start = 0
        self.end = 0

    def append(self, samples: numpy.ndarray) -> None:
        if len(samples):
            self._chunks.append(samples)
            self.end += len(samples)

    def take(self, sample_count: int) -> numpy.ndarray:
        """앞에서 sample_count개를 꺼낸다. 호출 전에 충분히 있는지 확인해야 한다."""
        if self.end - self.start < sample_count:
            raise ValueError("버퍼에 요청한 만큼의 샘플이 없습니다")
        pieces = []
        remaining = sample_count
        while remaining:
            head = self._chunks[0]
            if len(head) <= remaining:
                pieces.append(self._chunks.popleft())
                remaining -= len(head)
            else:
                pieces.append(head[:remaining])
                self._chunks[0] = head[remaining:]
                remaining = 0
        self.start += sample_count
        return numpy.concatenate(pieces) if pieces else numpy.zeros(0, numpy.int16)

    def skip_to(self, position: int) -> int:
        """position 앞의 샘플을 버리고 버린 개수를 반환한다. 아직 안 온 구간이면 위치만 옮긴다."""
        discard_count = min(max(position - self.start, 0), self.end - self.start)
        if discard_count:
            self.take(discard_count)
        self.end = max(self.end, position)
        self.start = max(self.start, position)
        return discard_count


@dataclass
class NetworkMicStats:
    gap_filled_chunks: int = 0  # seq가 건너뛰어 무음으로 채운 청크 (Pi 쪽 드롭)
    late_discarded_samples: int = 0  # 이미 지나간 tick 구간에 늦게 도착해 버린 샘플
    stalled_ticks: int = 0  # 마감을 넘겨 빠진 tick
    duplicate_chunks: int = 0  # 이미 받은 구간이 다시 온 청크
    overflow_chunks: int = 0  # Pi 콜백 overflow flag가 켜진 청크


@dataclass
class _MicState:
    room: str
    sample_rate: int
    hop_samples: int
    inbound: InboundBuffer = field(default_factory=InboundBuffer)
    stream_end: int = 0  # 지금까지 받은(무음 채움 포함) 스트림의 끝 위치
    last_boundary: int = 0  # 도착 시각을 기록한 마지막 tick 경계
    boundary_arrivals: dict[int, float] = field(default_factory=dict)
    stats: NetworkMicStats = field(default_factory=NetworkMicStats)


class NetworkStream:
    """연결 하나의 마이크들이 공유하는 수신 버퍼와 tick 결정 (스레드 안전).

    도착 시각은 time.perf_counter()로 잰다. Windows의 time.monotonic()은 약 15.6ms 단위라
    처리 지연(수십 ms) 측정에 거칠기 때문이다.

    시간축은 오디오 샘플 수다. seq × chunk_samples가 그 청크의 절대 위치이고, 빈 seq는 무음으로
    채운다. tick k는 마이크마다 [(k-1)·hop, k·hop) 구간을 소비한다.

    tick 결정 규칙:
    - 모든 마이크에 구간이 도착하면 바로 진행한다.
    - 일부만 도착했으면, 처음 도착한 마이크의 도착 시각 + stall_timeout까지 기다린 뒤
      도착한 마이크만으로 진행한다. 빠진 마이크의 그 구간은 나중에 와도 버린다.
    - 아무 마이크도 도착하지 않았으면 계속 기다린다(시간축이 오디오 기준이라 나중에 따라잡는다).
    - 연결이 닫히면 이미 도착한 구간까지만 처리하고 끝낸다.
    """

    def __init__(
        self,
        mics: list[tuple[int, str, int]],
        chunk_samples: int,
        hop_sec: float,
        stall_timeout_sec: float,
        backlog_max_sec: float,
    ) -> None:
        self.chunk_samples = chunk_samples
        self.hop_sec = hop_sec
        self._stall_timeout_sec = stall_timeout_sec
        self._backlog_max_sec = backlog_max_sec
        self._condition = threading.Condition()
        self._closed = False
        self._mics = {
            index: _MicState(room, sample_rate, round(hop_sec * sample_rate))
            for index, room, sample_rate in mics
        }
        self._decisions: dict[int, frozenset[int]] = {}
        self._ready_times: dict[int, float] = {}
        self.sources = [
            NetworkSource(self, index, room, sample_rate)
            for index, room, sample_rate in mics
        ]

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def has_mic(self, mic_index: int) -> bool:
        return mic_index in self._mics

    def stats(self) -> dict[str, NetworkMicStats]:
        with self._condition:
            return {state.room: state.stats for state in self._mics.values()}

    def receive(
        self, mic_index: int, seq: int, samples: numpy.ndarray, overflow: bool
    ) -> None:
        """수신 스레드가 AUDIO 청크를 넣는다. 처리 밀림이 상한을 넘으면 BacklogOverflowError."""
        with self._condition:
            state = self._mics[mic_index]
            if overflow:
                state.stats.overflow_chunks += 1
            position = seq * self.chunk_samples
            if position < state.stream_end:
                overlap = state.stream_end - position
                if overlap >= len(samples):
                    state.stats.duplicate_chunks += 1
                    return
                samples = samples[overlap:]
                position = state.stream_end
            if position > state.stream_end:
                gap = position - state.stream_end
                state.stats.gap_filled_chunks += gap // self.chunk_samples
                self._append(state, numpy.zeros(gap, dtype=numpy.int16))
            self._append(state, samples)
            self._record_boundary_arrivals(state, time.perf_counter())
            backlog_sec = (state.inbound.end - state.inbound.start) / state.sample_rate
            self._condition.notify_all()
        if backlog_sec > self._backlog_max_sec:
            raise BacklogOverflowError(
                f"{state.room}: 처리 대기 오디오 {backlog_sec:.1f}초가 "
                f"상한 {self._backlog_max_sec}초를 넘었습니다"
            )

    def _append(self, state: _MicState, samples: numpy.ndarray) -> None:
        start = state.stream_end
        state.stream_end += len(samples)
        consumed = state.inbound.start
        if start < consumed:
            discard_count = min(consumed - start, len(samples))
            state.stats.late_discarded_samples += discard_count
            samples = samples[discard_count:]
        state.inbound.append(samples)

    @staticmethod
    def _record_boundary_arrivals(state: _MicState, now: float) -> None:
        while (state.last_boundary + 1) * state.hop_samples <= state.stream_end:
            state.last_boundary += 1
            state.boundary_arrivals[state.last_boundary] = now

    def close(self) -> None:
        """연결 종료. 기다리는 tick 결정을 깨운다."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def decide_tick(self, tick: int) -> frozenset[int] | None:
        """tick에 포함할 마이크 index 집합. 연결이 닫혀 더 진행할 수 없으면 None."""
        with self._condition:
            while tick not in self._decisions:
                ready = frozenset(
                    index
                    for index, state in self._mics.items()
                    if state.stream_end >= tick * state.hop_samples
                )
                if len(ready) == len(self._mics) or (self._closed and ready):
                    self._record_decision(tick, ready)
                elif self._closed:
                    return None
                elif ready:
                    anchor = min(
                        self._mics[index].boundary_arrivals[tick] for index in ready
                    )
                    remaining = anchor + self._stall_timeout_sec - time.perf_counter()
                    if remaining <= 0:
                        self._record_decision(tick, ready)
                    else:
                        self._condition.wait(remaining)
                else:
                    self._condition.wait()
            return self._decisions[tick]

    def _record_decision(self, tick: int, ready: frozenset[int]) -> None:
        self._decisions[tick] = ready
        self._ready_times[tick] = max(
            self._mics[index].boundary_arrivals[tick] for index in ready
        )
        for index, state in self._mics.items():
            if index not in ready:
                state.stats.stalled_ticks += 1
                logging.getLogger(__name__).warning(
                    "%s: tick %d 마감(%.1f초) 초과로 제외",
                    state.room,
                    tick,
                    self._stall_timeout_sec,
                )
        # 모든 소스가 같은 tick을 조회한 뒤에는 필요 없으므로 오래된 기록을 지운다.
        for old_tick in [old for old in self._decisions if old < tick - 1]:
            del self._decisions[old_tick]
            self._ready_times.pop(old_tick, None)
        for state in self._mics.values():
            for old_tick in [old for old in state.boundary_arrivals if old < tick - 1]:
                del state.boundary_arrivals[old_tick]

    def take(self, mic_index: int, include: bool) -> numpy.ndarray | None:
        """tick 하나만큼 소비한다. 포함이면 그 구간 PCM(int16), 제외면 버리고 None."""
        with self._condition:
            state = self._mics[mic_index]
            if include:
                return state.inbound.take(state.hop_samples)
            end = state.inbound.start + state.hop_samples
            state.stats.late_discarded_samples += state.inbound.skip_to(end)
            return None

    def tick_ready_time(self, tick: int) -> float | None:
        """tick에 포함된 마이크의 데이터가 모두 도착한 time.perf_counter() 시각."""
        with self._condition:
            return self._ready_times.get(tick)


class NetworkSource:
    """NetworkStream의 마이크 하나를 AudioSource로 보여준다.

    advance(hop)는 그 tick 구간이 도착할 때까지(또는 마감까지) 기다렸다가 정확히 hop만큼
    링버퍼에 넣는다. 그래서 FrameProducer를 realtime_pacing=False로 돌리면 프레임 시각이
    오디오 샘플 수 기준이 된다.
    """

    def __init__(
        self, stream: NetworkStream, mic_index: int, name: str, sample_rate: int
    ) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.mic_index = mic_index
        self._stream = stream
        self._ring_buffer = RingBuffer(int(config.RING_BUFFER_SEC * sample_rate))
        self._tick = 0
        self._ended = False

    @property
    def overflow_count(self) -> int:
        return self._stream.stats()[self.name].overflow_chunks

    @property
    def exhausted(self) -> bool:
        return self._ended

    def start(self) -> None:
        """수신은 서버 수신 스레드가 하므로 할 일이 없다."""

    def advance(self, duration_sec: float) -> None:
        """다음 tick 구간을 기다렸다가 링버퍼에 넣는다 (side effect: 대기)."""
        if abs(duration_sec - self._stream.hop_sec) > 1e-9:
            raise ValueError(
                f"advance({duration_sec})가 스트림 hop {self._stream.hop_sec}과 다릅니다"
            )
        self._tick += 1
        included = self._stream.decide_tick(self._tick)
        if included is None:
            self._ended = True
            return
        samples = self._stream.take(self.mic_index, self.mic_index in included)
        if samples is not None:
            self._ring_buffer.write(samples.astype(numpy.float32) / PCM_FULL_SCALE)

    def read_new(self, last_total_written: int) -> tuple[numpy.ndarray, int]:
        return self._ring_buffer.read_new(last_total_written)

    def latest(self, num_samples: int) -> numpy.ndarray | None:
        return self._ring_buffer.latest(num_samples)

    def close(self) -> None:
        """스트림을 닫아 기다리는 advance를 깨운다."""
        self._stream.close()
