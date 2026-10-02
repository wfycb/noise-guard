"""CED 로딩과 배치 추론.

Phase 2에서는 CedClassifier.classify 를 같은 시그니처의 TCP 클라이언트로 교체한다.

torch·transformers는 CedClassifier를 실제로 만들 때만 import한다. main.py의 파이프라인을
가짜 분류기로 돌리는 테스트에서 수 초짜리 import를 피하기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

import numpy
from scipy.signal import resample_poly

import config
from label_map import AirborneScope, build_category_indices, category_probabilities
from models import Category

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ClassificationResult:
    probabilities: numpy.ndarray  # shape (라벨 수,), 0~1
    category_probs: dict[Category, float]  # 카테고리별 최대 확률
    top_labels: list[tuple[str, float]]  # 확률 내림차순 (라벨, 확률)


def resample_for_classifier(samples: numpy.ndarray, source_rate: int) -> numpy.ndarray:
    """수집 샘플레이트 → CED 입력 샘플레이트(16kHz)로 리샘플한다."""
    divisor = gcd(source_rate, config.CLASSIFIER_SAMPLE_RATE)
    upsample_factor = config.CLASSIFIER_SAMPLE_RATE // divisor
    downsample_factor = source_rate // divisor
    resampled = resample_poly(samples, upsample_factor, downsample_factor)
    return resampled.astype(numpy.float32)


def select_torch_device() -> torch.device:
    """CUDA → MPS → CPU 순으로 사용 가능한 장치를 고른다."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CedClassifier:
    """CED 모델 래퍼 (side effect: 생성 시 모델 다운로드/로딩)."""

    def __init__(self, airborne_scope: AirborneScope = config.AIRBORNE_SCOPE) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = select_torch_device()
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.CED_MODEL_NAME,
            revision=config.CED_MODEL_REVISION,
            trust_remote_code=True,
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
            config.CED_MODEL_NAME,
            revision=config.CED_MODEL_REVISION,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()
        self.id2label: dict[int, str] = dict(self._model.config.id2label)
        ordered_labels = [self.id2label[index] for index in range(len(self.id2label))]
        self._category_indices = build_category_indices(ordered_labels, airborne_scope)

    def predict_probabilities(self, windows: list[numpy.ndarray]) -> numpy.ndarray:
        """16kHz 창 여러 개를 한 번의 forward로 추론해 (창 수, 라벨 수) 확률을 반환한다."""
        import torch

        inputs = self._feature_extractor(
            windows,
            sampling_rate=config.CLASSIFIER_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model(**inputs)
        # CED 출력은 이미 sigmoid가 적용된 확률이므로 다시 적용하지 않는다.
        # 근거: modeling_ced.py의 CedForAudioClassification.forward_head가
        # `self.outputlayer(x).sigmoid()`를 반환하고, 그 값이 그대로 .logits에 담긴다.
        return output.logits.float().cpu().numpy()

    def classify(
        self, batch: dict[str, numpy.ndarray]
    ) -> dict[str, ClassificationResult]:
        """마이크 이름 → 16kHz 창. 모든 마이크를 한 번의 forward로 분류한다."""
        mic_names = list(batch)
        probabilities = self.predict_probabilities([batch[name] for name in mic_names])
        return {
            name: self.build_result(probabilities[index])
            for index, name in enumerate(mic_names)
        }

    def build_result(self, probabilities: numpy.ndarray) -> ClassificationResult:
        """라벨 확률 벡터 하나에서 카테고리별 확률과 top-k 라벨을 만든다."""
        top_indices = numpy.argsort(probabilities)[::-1][: config.TOP_K_LABELS]
        top_labels = [
            (self.id2label[int(index)], float(probabilities[index]))
            for index in top_indices
        ]
        return ClassificationResult(
            probabilities=probabilities,
            category_probs=category_probabilities(
                probabilities, self._category_indices
            ),
            top_labels=top_labels,
        )
