"""레벨 계산: dBFS, A-weighting, Fast(125ms) 블록 레벨, Leq, Lmax (순수 계산, 입출력 없음)."""

import numpy
from scipy.signal import bilinear, lfilter

import config


def rms_dbfs(samples: numpy.ndarray) -> float:
    """RMS를 dBFS로 변환한다. 무음이어도 유한한 값, 빈 배열은 ValueError."""
    if len(samples) == 0:
        raise ValueError("빈 배열의 dBFS는 정의되지 않습니다")
    rms = float(numpy.sqrt(numpy.mean(numpy.square(samples, dtype=numpy.float64))))
    return 20.0 * numpy.log10(rms + config.DBFS_EPSILON)


def leq_db(levels: numpy.ndarray | list[float]) -> float:
    """에너지 평균 레벨: 10*log10(mean(10^(L/10))). 빈 입력은 ValueError."""
    levels_array = numpy.asarray(levels, dtype=numpy.float64)
    if levels_array.size == 0:
        raise ValueError("빈 레벨 목록의 Leq는 정의되지 않습니다")
    return float(10.0 * numpy.log10(numpy.mean(10.0 ** (levels_array / 10.0))))


def a_weighting_coefficients(sample_rate: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """IEC 61672 아날로그 A특성 전달함수를 bilinear 변환한 디지털 IIR 계수 (b, a)."""
    omega_1 = 2.0 * numpy.pi * config.A_WEIGHTING_F1
    omega_2 = 2.0 * numpy.pi * config.A_WEIGHTING_F2
    omega_3 = 2.0 * numpy.pi * config.A_WEIGHTING_F3
    omega_4 = 2.0 * numpy.pi * config.A_WEIGHTING_F4
    # A1000은 1kHz에서 게인을 0 dB로 맞추기 위한 정규화 상수다.
    gain = omega_4**2 * 10.0 ** (config.A_WEIGHTING_A1000_DB / 20.0)
    numerator = [gain, 0.0, 0.0, 0.0, 0.0]
    denominator = numpy.polymul(
        numpy.polymul(
            [1.0, 2.0 * omega_4, omega_4**2], [1.0, 2.0 * omega_1, omega_1**2]
        ),
        numpy.polymul([1.0, omega_3], [1.0, omega_2]),
    )
    return bilinear(numerator, denominator, sample_rate)


class AWeightingFilter:
    """블록 경계에서 끊김이 없도록 필터 상태(zi)를 유지하는 A특성 필터."""

    def __init__(self, sample_rate: int) -> None:
        self._numerator, self._denominator = a_weighting_coefficients(sample_rate)
        # 장치 시작 시점의 입력은 알 수 없으므로 정지 상태(0)에서 시작한다.
        self._state = numpy.zeros(len(self._denominator) - 1)

    def process(self, samples: numpy.ndarray) -> numpy.ndarray:
        """블록을 필터링하고 다음 블록을 위해 상태를 갱신한다."""
        filtered, self._state = lfilter(
            self._numerator, self._denominator, samples, zi=self._state
        )
        return filtered


class LevelMeter:
    """A특성 필터 + Fast(125ms) 블록 분할. 블록을 못 채운 나머지는 다음 호출로 넘긴다."""

    def __init__(self, sample_rate: int) -> None:
        self._filter = AWeightingFilter(sample_rate)
        self._block_length = round(config.FAST_BLOCK_SEC * sample_rate)
        self._pending = numpy.zeros(0, dtype=numpy.float64)

    def push(self, samples: numpy.ndarray) -> list[float]:
        """새 샘플을 넣고, 이번에 완성된 125ms 블록들의 dBFS(A)를 시간순으로 반환한다."""
        weighted = self._filter.process(samples.astype(numpy.float64))
        self._pending = numpy.concatenate((self._pending, weighted))
        block_count = len(self._pending) // self._block_length
        used_length = block_count * self._block_length
        blocks = self._pending[:used_length].reshape(block_count, self._block_length)
        levels = [rms_dbfs(block) for block in blocks]
        self._pending = self._pending[used_length:]
        return levels


def to_estimated_dba(dbfs_a: float, mic_name: str) -> float:
    """dBFS(A) → 추정 dB(A). 마이크별 보정 오프셋이 없으면 기본 오프셋을 쓴다."""
    offset = config.CALIBRATION_OFFSET_DB.get(
        mic_name, config.DEFAULT_CALIBRATION_OFFSET_DB
    )
    return dbfs_a + offset
