"""알림이 나면 앞뒤 오디오(모든 마이크)와 메타데이터를 저장한다. 판단에는 쓰지 않는다.

목적: 미니어처 실측 데이터를 모아 임계값·오프셋을 다시 조정하는 것. 수집은 멈추지 않는다.
메인 스레드는 tick마다 on_tick으로 오디오를 넘기기만 하고, 파일 쓰기와 용량 정리는 저장 스레드가 한다.

저장 형식: data/events/YYYYMMDD_HHMMSS_<규칙>/
    <방>.wav     원래 샘플레이트, int16, 알림 시각 기준 앞 EVENT_PRE_SEC + 뒤 EVENT_POST_SEC
    meta.json    합친 알림, 대표 마이크, 프레임별 마이크 측정값, 보정 오프셋, 모델, 임계값
"""

import json
import logging
import queue
import shutil
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy
from scipy.io import wavfile

import config
from alert import merge_alerts
from decision import LOCAL_TIMEZONE
from models import Alert, Frame, MicMeasurement

logger = logging.getLogger(__name__)

INT16_FULL_SCALE = 32768.0
BYTES_PER_MEGABYTE = 1024 * 1024


@dataclass
class _Recording:
    """진행 중인 녹음. 시각은 오디오 시간축(프레임 timestamp) 기준이다."""

    start_timestamp: float
    end_timestamp: float
    first_alert_timestamp: float
    folder_rules: str
    audio: dict[str, list[numpy.ndarray]]
    collected: dict[str, int]
    alerts: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    pre_padded_sec: float = 0.0  # 앞부분 오디오가 모자라 무음으로 채운 길이


@dataclass(frozen=True)
class EventToSave:
    folder_name: str
    audio: dict[str, numpy.ndarray]  # float32, 원래 샘플레이트
    sample_rates: dict[str, int]
    meta: dict[str, Any]


def frame_record(
    frame: Frame, measurements: dict[str, MicMeasurement]
) -> dict[str, Any]:
    """meta.json에 넣을 프레임 하나 (순수 함수)."""
    return {
        "timestamp": frame.timestamp,
        "local_time": datetime.fromtimestamp(frame.timestamp, LOCAL_TIMEZONE).isoformat(
            timespec="seconds"
        ),
        "representative_mic": frame.mic_name,
        "mics": {
            room: {
                "leq_db": round(measurement.leq_db, 2),
                "lmax_db": round(measurement.lmax_db, 2),
                "category_probs": {
                    category.value: round(probability, 4)
                    for category, probability in measurement.category_probs.items()
                },
                "top_label": measurement.top_label,
            }
            for room, measurement in measurements.items()
        },
    }


def alert_record(alerts: list[Alert], timestamp: float) -> dict[str, Any]:
    """같은 프레임의 알림들을 합친 기록 (순수 함수). 원본 규칙별 내용도 함께 둔다."""
    merged = merge_alerts(alerts)
    assert merged is not None
    return {
        "timestamp": timestamp,
        "level": merged.level,
        "rules": list(merged.rules),
        "room": alerts[0].mic_name,
        "messages": list(merged.messages),
        "peak_db": round(max(alert.peak_db for alert in alerts), 2),
        "count": max(alert.count for alert in alerts),
        "room_counts": next(
            (alert.room_counts for alert in alerts if alert.room_counts), {}
        ),
    }


class EventRecorder:
    """알림 전후 오디오를 모아 저장 스레드에 넘긴다.

    on_tick은 메인 스레드에서 tick마다 시간 순서대로 불러야 한다. 마이크가 빠진 tick은 무음으로 채워
    마이크 사이의 시간이 어긋나지 않게 한다.
    """

    def __init__(
        self,
        output_dir: Path,
        context: dict[str, Any],
        pre_sec: float = config.EVENT_PRE_SEC,
        post_sec: float = config.EVENT_POST_SEC,
        max_sec: float = config.EVENT_MAX_SEC,
        max_total_mb: float = config.EVENT_MAX_TOTAL_MB,
    ) -> None:
        self._output_dir = output_dir
        self._context = (
            context  # 모델 이름, 임계값, 게이트 등 meta.json에 그대로 넣을 값
        )
        self._pre_sec = pre_sec
        self._post_sec = post_sec
        self._max_sec = max_sec
        self._max_total_mb = max_total_mb
        self._sample_rates: dict[str, int] = {}
        self._history: dict[str, deque[numpy.ndarray]] = {}
        self._recent_frames: deque[dict[str, Any]] = deque()
        self._recording: _Recording | None = None
        self._save_queue: queue.Queue[EventToSave | None] = queue.Queue()
        self._saver = threading.Thread(
            target=self._save_loop,
            name=f"{config.THREAD_NAME_PREFIX}-event-saver",
            daemon=True,
        )
        self._saver.start()
        self.saved_folders: list[Path] = []
        self.deleted_folders: list[str] = []

    def start_stream(self, sample_rates: dict[str, int]) -> None:
        """새 입력(연결)이 시작됐다. 진행 중이던 녹음은 잘린 채로 저장하고 버퍼를 비운다."""
        self._finish_recording(truncated=True)
        self._sample_rates = dict(sample_rates)
        self._history = {room: deque() for room in sample_rates}
        self._recent_frames.clear()

    def on_tick(
        self,
        timestamp: float,
        audio: dict[str, numpy.ndarray],
        frame: Frame | None,
        measurements: dict[str, MicMeasurement],
        alerts: list[Alert],
    ) -> None:
        """tick 하나(끝 시각 timestamp)의 오디오·프레임·알림을 반영한다. 파일 쓰기는 하지 않는다."""
        chunks = self._aligned_chunks(audio)
        self._remember(chunks)
        frame_info = frame_record(frame, measurements) if frame is not None else None
        if frame_info is not None:
            self._recent_frames.append(frame_info)
            while self._recent_frames and (
                self._recent_frames[0]["timestamp"] <= timestamp - self._pre_sec
            ):
                self._recent_frames.popleft()
        recording = self._recording
        if recording is not None:
            for room, chunk in chunks.items():
                recording.audio[room].append(chunk)
                recording.collected[room] += len(chunk)
            if frame_info is not None:
                recording.frames.append(frame_info)
        if alerts:
            self._on_alert(timestamp, alerts)
        if self._recording is not None and self._recording_complete():
            self._finish_recording(truncated=False)

    def close(self) -> None:
        """진행 중인 녹음을 저장하고 저장 스레드를 끝낸다 (side effect: 대기)."""
        self._finish_recording(truncated=True)
        self._save_queue.put(None)
        self._saver.join(config.THREAD_JOIN_TIMEOUT_SEC)

    def _aligned_chunks(
        self, audio: dict[str, numpy.ndarray]
    ) -> dict[str, numpy.ndarray]:
        hop_sec = config.CLASSIFY_HOP_SEC
        chunks = {}
        for room, sample_rate in self._sample_rates.items():
            chunk = audio.get(room)
            if chunk is None or len(chunk) == 0:
                chunk = numpy.zeros(round(hop_sec * sample_rate), dtype=numpy.float32)
            chunks[room] = chunk
        return chunks

    def _remember(self, chunks: dict[str, numpy.ndarray]) -> None:
        """앞부분용으로 최근 EVENT_PRE_SEC보다 조금 넉넉히(한 tick 더) 오디오를 기억한다."""
        for room, chunk in chunks.items():
            history = self._history[room]
            history.append(chunk)
            keep = round(
                (self._pre_sec + config.CLASSIFY_HOP_SEC) * self._sample_rates[room]
            )
            while sum(len(item) for item in history) - len(history[0]) >= keep:
                history.popleft()

    def _on_alert(self, timestamp: float, alerts: list[Alert]) -> None:
        record = alert_record(alerts, timestamp)
        if self._recording is None:
            start = timestamp - self._pre_sec
            pre_audio = {}
            pre_padded_sec = 0.0
            for room, sample_rate in self._sample_rates.items():
                pre_samples = round(self._pre_sec * sample_rate)
                joined = numpy.concatenate(list(self._history[room]))[-pre_samples:]
                # 시작 직후라 앞 오디오가 모자라면 무음으로 채워 녹음 구간의 시각이 어긋나지 않게 한다.
                missing = pre_samples - len(joined)
                if missing > 0:
                    pre_padded_sec = max(pre_padded_sec, missing / sample_rate)
                    joined = numpy.concatenate(
                        (numpy.zeros(missing, dtype=numpy.float32), joined)
                    )
                pre_audio[room] = joined
            self._recording = _Recording(
                start_timestamp=start,
                end_timestamp=min(timestamp + self._post_sec, start + self._max_sec),
                first_alert_timestamp=timestamp,
                folder_rules="+".join(record["rules"]),
                audio={room: [samples] for room, samples in pre_audio.items()},
                collected={room: len(samples) for room, samples in pre_audio.items()},
                frames=list(self._recent_frames),
                pre_padded_sec=pre_padded_sec,
            )
            logger.info("이벤트 녹음 시작 (%s)", self._recording.folder_rules)
        else:
            # 녹음 중 새 알림: 끝 시각만 늦춘다. 전체 길이는 EVENT_MAX_SEC를 넘지 않는다.
            recording = self._recording
            recording.end_timestamp = min(
                max(recording.end_timestamp, timestamp + self._post_sec),
                recording.start_timestamp + self._max_sec,
            )
        self._recording.alerts.append(record)

    def _target_samples(self, room: str) -> int:
        recording = self._recording
        assert recording is not None
        duration = recording.end_timestamp - recording.start_timestamp
        return round(duration * self._sample_rates[room])

    def _recording_complete(self) -> bool:
        recording = self._recording
        assert recording is not None
        return all(
            recording.collected[room] >= self._target_samples(room)
            for room in self._sample_rates
        )

    def _finish_recording(self, truncated: bool) -> None:
        recording = self._recording
        if recording is None:
            return
        audio = {
            room: numpy.concatenate(chunks)[: self._target_samples(room)].astype(
                numpy.float32
            )
            for room, chunks in recording.audio.items()
        }
        self._recording = None
        first_time = datetime.fromtimestamp(
            recording.first_alert_timestamp, LOCAL_TIMEZONE
        )
        representative = recording.alerts[0]["room"]
        meta = {
            "first_alert_time": first_time.isoformat(timespec="seconds"),
            "start_timestamp": recording.start_timestamp,
            "end_timestamp": recording.end_timestamp,
            "pre_sec": self._pre_sec,
            "post_sec": self._post_sec,
            "duration_sec": round(
                min(
                    len(audio[room]) / rate for room, rate in self._sample_rates.items()
                ),
                3,
            ),
            "truncated": truncated,
            "pre_padded_sec": round(recording.pre_padded_sec, 3),
            "representative_mic": representative,
            "alerts": recording.alerts,
            "frames": [
                frame
                for frame in recording.frames
                if recording.start_timestamp
                < frame["timestamp"]
                <= recording.end_timestamp
            ],
            "sample_rates": dict(self._sample_rates),
            "calibration": {
                room: {
                    "offset_db": config.CALIBRATION_OFFSET_DB.get(
                        room, config.DEFAULT_CALIBRATION_OFFSET_DB
                    ),
                    "calibrated": room in config.CALIBRATION_OFFSET_DB,
                }
                for room in self._sample_rates
            },
            **self._context,
        }
        folder_name = f"{first_time:%Y%m%d_%H%M%S}_{recording.folder_rules}"
        self._save_queue.put(
            EventToSave(folder_name, audio, dict(self._sample_rates), meta)
        )

    def _save_loop(self) -> None:
        while True:
            event = self._save_queue.get()
            if event is None:
                return
            try:
                folder = save_event(event, self._output_dir)
            except OSError as error:
                # 디스크 오류가 나도 수집·판단은 계속돼야 하므로 로그만 남긴다.
                logger.error("이벤트 저장 실패 %s — %s", event.folder_name, error)
                continue
            deleted = enforce_total_size(
                self._output_dir, self._max_total_mb, keep=folder
            )
            self.deleted_folders.extend(deleted)
            self.saved_folders.append(folder)
            logger.info("이벤트 저장: %s", folder)


def save_event(event: EventToSave, output_dir: Path) -> Path:
    """이벤트 폴더에 마이크별 wav(int16)와 meta.json을 쓴다 (side effect: 파일 쓰기)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = output_dir / event.folder_name
    suffix = 2
    while folder.exists():
        folder = output_dir / f"{event.folder_name}_{suffix}"
        suffix += 1
    folder.mkdir()
    for room, samples in event.audio.items():
        pcm = numpy.clip(
            numpy.round(samples * INT16_FULL_SCALE),
            -INT16_FULL_SCALE,
            INT16_FULL_SCALE - 1,
        ).astype(numpy.int16)
        wavfile.write(folder / f"{room}.wav", event.sample_rates[room], pcm)
    (folder / "meta.json").write_text(
        json.dumps(event.meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder


def folder_size_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def enforce_total_size(output_dir: Path, max_total_mb: float, keep: Path) -> list[str]:
    """전체 용량이 max_total_mb를 넘으면 가장 오래된 이벤트부터 지운다 (side effect: 파일 삭제, 로그).

    폴더 이름이 시각으로 시작하므로 이름 순서가 오래된 순서다. 방금 저장한 폴더(keep)는 지우지 않는다.
    """
    limit_bytes = max_total_mb * BYTES_PER_MEGABYTE
    folders = sorted(path for path in output_dir.iterdir() if path.is_dir())
    sizes = {folder: folder_size_bytes(folder) for folder in folders}
    total = sum(sizes.values())
    deleted = []
    for folder in folders:
        if total <= limit_bytes:
            break
        if folder == keep:
            continue
        shutil.rmtree(folder)
        total -= sizes[folder]
        deleted.append(folder.name)
        logger.warning(
            "이벤트 용량 %.1f MB 초과로 가장 오래된 이벤트 삭제: %s (%.1f MB)",
            max_total_mb,
            folder.name,
            sizes[folder] / BYTES_PER_MEGABYTE,
        )
    return deleted
