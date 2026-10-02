"""조용할 때 CED 건너뛰기 게이트 테스트 (모델 없이 가짜 분류기 사용)."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import pytest
from scipy.io import wavfile

import config
from capture import FileSource
from classifier import judgment_limits_db, validate_skip_gate
from decision import DecisionConfig, NoiseDecisionEngine
from models import Category

if TYPE_CHECKING:
    from classifier import ClassificationResult
    from main import RunResult

SAMPLE_RATE = 48000


class RecordingClassifier:
    """받은 배치의 마이크 이름을 기록하고 IMPACT 0.9를 돌려주는 가짜 분류기."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def classify(
        self, batch: dict[str, numpy.ndarray]
    ) -> dict[str, ClassificationResult]:
        from classifier import ClassificationResult

        self.batches.append(sorted(batch))
        probs = {category: 0.0 for category in Category}
        probs[Category.IMPACT] = 0.9
        return {
            name: ClassificationResult(numpy.zeros(1), dict(probs), [("Stub", 0.9)])
            for name in batch
        }


def write_wav(path: Path, amplitude: float, duration_sec: float = 4.0) -> Path:
    time_axis = numpy.arange(int(duration_sec * SAMPLE_RATE)) / SAMPLE_RATE
    wavfile.write(
        path,
        SAMPLE_RATE,
        (amplitude * numpy.sin(2 * numpy.pi * 1000 * time_axis)).astype(numpy.float32),
    )
    return path


def run_files(
    files: dict[str, Path], classifier: RecordingClassifier, gate_db: float | None
) -> RunResult:
    from main import run_sources

    return run_sources(
        [FileSource(room, path, end_behavior="stop") for room, path in files.items()],
        classifier,
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        {
            "start_timestamp": 0.0,
            "realtime_pacing": False,
            "stop_when_exhausted": True,
            "duration_sec": None,
            "skip_quiet_below_db": gate_db,
        },
        None,
        None,
    )


def test_default_gate_satisfies_margin() -> None:
    validate_skip_gate(
        config.SKIP_CLASSIFY_BELOW_DB, config.SKIP_MARGIN_DB, judgment_limits_db()
    )
    assert min(judgment_limits_db()) == config.IMPACT_LEQ_LIMIT_NIGHT_DB


def test_gate_at_limit_minus_margin_is_allowed() -> None:
    validate_skip_gate(24.0, 10.0, [34.0, 52.0])


def test_gate_above_limit_minus_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="24"):
        validate_skip_gate(24.1, 10.0, [34.0, 52.0])


def test_producer_refuses_unsafe_gate_at_start() -> None:
    from main import FrameProducer

    with pytest.raises(ValueError):
        FrameProducer(
            [], RecordingClassifier(), None, 0.0, False, True, None, None, 30.0
        )


def test_only_loud_mic_goes_to_classifier(tmp_path: Path) -> None:
    classifier = RecordingClassifier()
    result = run_files(
        {
            "거실": write_wav(tmp_path / "loud.wav", 0.5),
            "안방": write_wav(tmp_path / "quiet.wav", 0.0),
        },
        classifier,
        gate_db=config.SKIP_CLASSIFY_BELOW_DB,
    )
    assert classifier.batches and all(batch == ["거실"] for batch in classifier.batches)
    assert result.skipped_mic_frames == len(result.frames)  # 매 프레임 안방 1개씩
    assert result.mic_frames == 2 * len(result.frames)


def test_quiet_mic_gets_empty_classification(tmp_path: Path) -> None:
    from main import FrameProducer, ProducedFrame

    output: queue.Queue = queue.Queue()
    source = FileSource("안방", write_wav(tmp_path / "quiet.wav", 0.0), "stop")
    producer = FrameProducer(
        [source],
        RecordingClassifier(),
        output,
        0.0,
        False,
        True,
        None,
        None,
        config.SKIP_CLASSIFY_BELOW_DB,
    )
    producer.run()  # 스레드를 띄우지 않고 이 스레드에서 끝까지 돌린다
    frames = []
    while not output.empty():
        item = output.get_nowait()
        if isinstance(item, ProducedFrame):
            frames.append(item)
    assert frames
    for item in frames:
        assert item.inference_ms is None
        assert item.frame.top_label == config.QUIET_TOP_LABEL
        assert all(prob == 0.0 for prob in item.frame.category_probs.values())


def test_all_quiet_never_calls_classifier(tmp_path: Path) -> None:
    classifier = RecordingClassifier()
    result = run_files(
        {
            "거실": write_wav(tmp_path / "a.wav", 0.0),
            "안방": write_wav(tmp_path / "b.wav", 0.0),
        },
        classifier,
        gate_db=config.SKIP_CLASSIFY_BELOW_DB,
    )
    assert classifier.batches == []
    assert result.frames and result.inference_ms == []
    assert result.skipped_mic_frames == result.mic_frames


def test_no_skip_classifies_everything(tmp_path: Path) -> None:
    classifier = RecordingClassifier()
    run_files(
        {
            "거실": write_wav(tmp_path / "a.wav", 0.5),
            "안방": write_wav(tmp_path / "b.wav", 0.0),
        },
        classifier,
        gate_db=None,
    )
    assert classifier.batches and all(
        batch == ["거실", "안방"] for batch in classifier.batches
    )


def test_frame_right_after_loud_sound_is_still_classified(tmp_path: Path) -> None:
    # 2.0~2.5초에만 소리. tick 4(3~4초)는 1초 Leq로는 조용하지만 분류 창(2~4초)에 소리가 있으므로
    # CED가 IMPACT로 볼 수 있다. 이 프레임을 건너뛰면 이벤트 병합이 --no-skip과 달라진다.
    time_axis = numpy.arange(6 * SAMPLE_RATE) / SAMPLE_RATE
    signal = numpy.where(
        (time_axis >= 2.0) & (time_axis < 2.5),
        0.5 * numpy.sin(2 * numpy.pi * 1000 * time_axis),
        0.0,
    ).astype(numpy.float32)
    path = tmp_path / "burst.wav"
    wavfile.write(path, SAMPLE_RATE, signal)
    result = run_files(
        {"거실": path}, RecordingClassifier(), gate_db=config.SKIP_CLASSIFY_BELOW_DB
    )
    labels = [frame.top_label for frame in result.frames]  # tick 2~6
    quiet = config.QUIET_TOP_LABEL
    assert labels == [quiet, "Stub", "Stub", quiet, quiet]
