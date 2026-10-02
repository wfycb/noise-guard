"""Pi 쪽 오디오 입력: 마이크(MicStream)와 테스트용 wav 재생(FileStream), 송신 큐.

scipy를 쓸 수 없으므로 wav는 numpy로 직접 읽는다(read_wav_int16).
"""

import logging
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy

from pi_client import client_config
from protocol import FLAG_OVERFLOW, MicInfo

logger = logging.getLogger(__name__)

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# EXTENSIBLE의 SubFormat GUID는 앞 2바이트가 포맷 코드이고 나머지 14바이트는 고정값이다.
KSDATAFORMAT_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
FMT_BASE = struct.Struct("<HHIIHH")
FMT_EXTENSIBLE = struct.Struct(
    "<HHIH14s"
)  # cbSize, validBits, channelMask, subformat code, GUID 나머지
CHUNK_HEADER = struct.Struct("<4sI")
INT16_FULL_SCALE = 32768.0
INT16_MIN = -32768
INT16_MAX = 32767


class UnsupportedWavError(ValueError):
    """읽을 수 없는 wav 형식."""


@dataclass(frozen=True)
class WavAudio:
    samples: numpy.ndarray  # int16 모노
    sample_rate: int
    clipped_samples: int  # float → int16 변환에서 [-1, 1] 밖이라 잘린 샘플 수


def parse_wav(data: bytes) -> WavAudio:
    """wav 바이트를 int16 모노로 읽는다 (순수 함수).

    지원: PCM16, float32, WAVE_FORMAT_EXTENSIBLE(하위 포맷이 PCM16·float32). 여러 채널은 평균한다.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise UnsupportedWavError("RIFF/WAVE 파일이 아닙니다")
    fmt_body: bytes | None = None
    pcm_body: bytes | None = None
    position = 12
    while position + CHUNK_HEADER.size <= len(data):
        chunk_id, chunk_size = CHUNK_HEADER.unpack_from(data, position)
        body_start = position + CHUNK_HEADER.size
        body = data[body_start : body_start + chunk_size]
        if chunk_id == b"fmt ":
            fmt_body = body
        elif chunk_id == b"data":
            pcm_body = body
        # RIFF 청크는 2바이트 정렬이라 크기가 홀수면 패딩 1바이트가 붙는다.
        position = body_start + chunk_size + (chunk_size & 1)
    if fmt_body is None or pcm_body is None:
        raise UnsupportedWavError("fmt 또는 data 청크가 없습니다")
    if len(fmt_body) < FMT_BASE.size:
        raise UnsupportedWavError("fmt 청크가 너무 짧습니다")
    format_tag, channels, sample_rate, _, block_align, bits = FMT_BASE.unpack_from(
        fmt_body
    )
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt_body) < FMT_BASE.size + FMT_EXTENSIBLE.size:
            raise UnsupportedWavError("EXTENSIBLE fmt 청크가 너무 짧습니다")
        _, _, _, format_tag, guid_tail = FMT_EXTENSIBLE.unpack_from(
            fmt_body, FMT_BASE.size
        )
        if guid_tail != KSDATAFORMAT_GUID_TAIL:
            raise UnsupportedWavError("EXTENSIBLE SubFormat GUID를 알 수 없습니다")
    if channels < 1 or block_align < 1:
        raise UnsupportedWavError(
            f"채널 수 {channels}, block_align {block_align}이 잘못됐습니다"
        )
    pcm_body = pcm_body[: len(pcm_body) - len(pcm_body) % block_align]

    if (format_tag, bits) == (WAVE_FORMAT_PCM, 16):
        frames = numpy.frombuffer(pcm_body, dtype="<i2").reshape(-1, channels)
        if channels == 1:
            return WavAudio(frames[:, 0].astype(numpy.int16), sample_rate, 0)
        mono = numpy.round(frames.mean(axis=1))
        return WavAudio(mono.astype(numpy.int16), sample_rate, 0)
    if (format_tag, bits) == (WAVE_FORMAT_IEEE_FLOAT, 32):
        frames = numpy.frombuffer(pcm_body, dtype="<f4").reshape(-1, channels)
        mono = frames.mean(axis=1, dtype=numpy.float64)
        clipped = int(numpy.count_nonzero(numpy.abs(mono) > 1.0))
        scaled = numpy.round(numpy.clip(mono, -1.0, 1.0) * INT16_FULL_SCALE)
        samples = numpy.clip(scaled, INT16_MIN, INT16_MAX).astype(numpy.int16)
        return WavAudio(samples, sample_rate, clipped)
    raise UnsupportedWavError(
        f"지원하지 않는 형식: format 0x{format_tag:04x}, {bits}bit "
        "(PCM16, float32, EXTENSIBLE의 PCM16/float32만 지원)"
    )


def read_wav_int16(path: Path) -> WavAudio:
    """wav 파일을 int16 모노로 읽는다 (side effect: 파일 읽기)."""
    return parse_wav(path.read_bytes())


@dataclass(frozen=True)
class AudioChunkItem:
    mic_index: int
    seq: int
    flags: int
    samples: numpy.ndarray  # int16
    capture_time: float  # time.monotonic(), 청크 끝 시점


class ChunkQueue:
    """크기가 제한된 송신 큐 (스레드 안전).

    마이크 콜백은 put_drop_oldest로 절대 막히지 않고, 파일 고속 재생은 put_blocking으로 기다린다.
    """

    def __init__(self, max_chunks: int) -> None:
        self._items: deque[AudioChunkItem] = deque()
        self._max_chunks = max_chunks
        self._condition = threading.Condition()
        self.dropped_chunks = 0

    def put_drop_oldest(self, item: AudioChunkItem) -> None:
        with self._condition:
            if len(self._items) >= self._max_chunks:
                self._items.popleft()
                self.dropped_chunks += 1
            self._items.append(item)
            self._condition.notify_all()

    def put_blocking(self, item: AudioChunkItem, stop_event: threading.Event) -> bool:
        """자리가 날 때까지 기다린다. stop_event가 켜지면 False."""
        with self._condition:
            while len(self._items) >= self._max_chunks:
                if stop_event.is_set():
                    return False
                self._condition.wait(client_config.QUEUE_POLL_SEC)
            self._items.append(item)
            self._condition.notify_all()
            return True

    def get(self, timeout: float) -> AudioChunkItem | None:
        with self._condition:
            if not self._items:
                self._condition.wait(timeout)
            if not self._items:
                return None
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def clear(self) -> int:
        """남은 청크를 버리고 개수를 반환한다 (재접속 시 끊긴 동안의 오디오 버리기)."""
        with self._condition:
            count = len(self._items)
            self._items.clear()
            self._condition.notify_all()
            return count

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


class FileStream:
    """wav 여러 개를 마이크처럼 청크 단위로 내보낸다 (테스트용, side effect: 파일 읽기·스레드).

    실시간 속도로 내보내고, fast면 기다리지 않는다. 길이가 다른 파일은 가장 긴 파일까지 무음으로
    채운다. 서버가 모든 마이크의 tick이 찰 때까지 기다리기 때문이다.
    """

    def __init__(self, files: dict[str, Path], chunk_samples: int, fast: bool) -> None:
        self._chunk_samples = chunk_samples
        self._fast = fast
        self._audio = {room: read_wav_int16(path) for room, path in files.items()}
        for room, audio in self._audio.items():
            if audio.clipped_samples:
                logger.warning(
                    "%s: float → int16 변환에서 %d샘플이 [-1, 1] 밖이라 잘렸습니다",
                    room,
                    audio.clipped_samples,
                )
            logger.info(
                "%s: %s, %d Hz, %.1f초",
                room,
                files[room],
                audio.sample_rate,
                len(audio.samples) / audio.sample_rate,
            )
        longest_sec = max(
            len(audio.samples) / audio.sample_rate for audio in self._audio.values()
        )
        self._chunk_counts = [
            int(numpy.ceil(longest_sec * audio.sample_rate / chunk_samples))
            for audio in self._audio.values()
        ]
        self.finished = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def mics(self) -> tuple[MicInfo, ...]:
        return tuple(
            MicInfo(index, room, audio.sample_rate)
            for index, (room, audio) in enumerate(self._audio.items())
        )

    @property
    def overflow_count(self) -> int:
        return 0

    def start(self, chunk_queue: ChunkQueue, stop_event: threading.Event) -> None:
        """재생 스레드를 시작한다 (side effect: 스레드)."""
        self._thread = threading.Thread(
            target=self._run,
            args=(chunk_queue, stop_event),
            name=f"{client_config.THREAD_NAME_PREFIX}-file-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(client_config.THREAD_JOIN_TIMEOUT_SEC)

    def _chunk(self, mic_index: int, seq: int) -> numpy.ndarray:
        samples = list(self._audio.values())[mic_index].samples
        start = seq * self._chunk_samples
        chunk = samples[start : start + self._chunk_samples]
        if len(chunk) < self._chunk_samples:
            chunk = numpy.concatenate(
                (chunk, numpy.zeros(self._chunk_samples - len(chunk), numpy.int16))
            )
        return chunk

    def _run(self, chunk_queue: ChunkQueue, stop_event: threading.Event) -> None:
        sample_rates = [audio.sample_rate for audio in self._audio.values()]
        next_seq = [0] * len(sample_rates)
        start_time = time.monotonic()
        try:
            while not stop_event.is_set():
                pending = [
                    index
                    for index, count in enumerate(self._chunk_counts)
                    if next_seq[index] < count
                ]
                if not pending:
                    return
                # 마이크마다 청크 길이(초)가 다를 수 있으므로, 청크 끝 시각이 가장 이른 것부터 보낸다.
                mic_index = min(
                    pending,
                    key=lambda index: (next_seq[index] + 1) / sample_rates[index],
                )
                seq = next_seq[mic_index]
                due = (
                    start_time
                    + (seq + 1) * self._chunk_samples / sample_rates[mic_index]
                )
                if not self._fast and stop_event.wait(max(0.0, due - time.monotonic())):
                    return
                item = AudioChunkItem(
                    mic_index, seq, 0, self._chunk(mic_index, seq), time.monotonic()
                )
                if self._fast:
                    if not chunk_queue.put_blocking(item, stop_event):
                        return
                else:
                    chunk_queue.put_drop_oldest(item)
                next_seq[mic_index] += 1
        finally:
            self.finished.set()


class MicStream:
    """마이크마다 InputStream을 열어 둔다 (side effect: 오디오 장치 접근).

    100ms마다 sd.rec을 다시 부르지 않고 스트림을 계속 열어 두므로 청크 사이에 빈틈이 없다.
    콜백은 int16 청크를 송신 큐에 넣기만 한다(가득 차면 가장 오래된 청크를 버린다).
    """

    def __init__(
        self, devices: dict[str, int | str], sample_rate: int, chunk_samples: int
    ) -> None:
        import sounddevice  # Pi에서 마이크 모드일 때만 필요하다(파일 테스트는 장치 없이).

        self._sounddevice = sounddevice
        self._devices = devices
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._streams: list[object] = []
        self._next_seq = [0] * len(devices)
        self._overflow_counts = [0] * len(devices)
        self.finished = threading.Event()  # 마이크는 끝나지 않는다.

    @property
    def mics(self) -> tuple[MicInfo, ...]:
        return tuple(
            MicInfo(index, room, self._sample_rate)
            for index, room in enumerate(self._devices)
        )

    @property
    def overflow_count(self) -> int:
        return sum(self._overflow_counts)

    def start(self, chunk_queue: ChunkQueue, stop_event: threading.Event) -> None:
        """장치를 열고 수집을 시작한다 (side effect: 장치 접근)."""
        for mic_index, device in enumerate(self._devices.values()):
            stream = self._sounddevice.InputStream(
                device=device,
                channels=1,
                samplerate=self._sample_rate,
                dtype="int16",
                blocksize=self._chunk_samples,
                callback=self._make_callback(mic_index, chunk_queue),
            )
            stream.start()
            self._streams.append(stream)

    def _make_callback(
        self, mic_index: int, chunk_queue: ChunkQueue
    ) -> Callable[[numpy.ndarray, int, object, object], None]:
        def on_audio_block(
            input_data: numpy.ndarray,
            frame_count: int,
            time_info: object,
            status: object,
        ) -> None:
            flags = 0
            if getattr(status, "input_overflow", False):
                self._overflow_counts[mic_index] += 1
                flags |= FLAG_OVERFLOW
            # seq는 큐에서 버려지는 청크까지 포함해 증가시킨다. 서버가 빈 구간을 알 수 있게 하기 위해서다.
            seq = self._next_seq[mic_index]
            self._next_seq[mic_index] += 1
            chunk_queue.put_drop_oldest(
                AudioChunkItem(
                    mic_index, seq, flags, input_data[:, 0].copy(), time.monotonic()
                )
            )

        return on_audio_block

    def reset_sequence(self) -> None:
        """재접속하면 새 스트림이므로 seq를 0부터 다시 센다."""
        self._next_seq = [0] * len(self._devices)

    def stop(self) -> None:
        """스트림을 닫는다 (side effect: 장치 해제)."""
        for stream in self._streams:
            stream.stop()
            stream.close()
        self._streams.clear()
