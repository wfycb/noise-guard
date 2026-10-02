import numpy
import pytest

import config
from level import (
    AWeightingFilter,
    LevelMeter,
    leq_db,
    rms_dbfs,
    to_estimated_dba,
)

SAMPLE_RATE = 48000
# 필터 과도응답을 버리고 정상상태만 비교하기 위한 구간 (100Hz도 수십 주기 지난 뒤).
SETTLE_SEC = 0.5


def sine(
    frequency: float, amplitude: float = 1.0, duration_sec: float = 1.0
) -> numpy.ndarray:
    time_axis = numpy.arange(int(duration_sec * SAMPLE_RATE)) / SAMPLE_RATE
    return amplitude * numpy.sin(2.0 * numpy.pi * frequency * time_axis)


def a_weighting_gain_db(frequency: float) -> float:
    signal = sine(frequency, duration_sec=SETTLE_SEC + 1.0)
    filtered = AWeightingFilter(SAMPLE_RATE).process(signal)
    settle_samples = int(SETTLE_SEC * SAMPLE_RATE)
    return rms_dbfs(filtered[settle_samples:]) - rms_dbfs(signal[settle_samples:])


def test_full_scale_sine_is_minus_3_dbfs() -> None:
    assert rms_dbfs(sine(1000.0)) == pytest.approx(-3.01, abs=0.05)


def test_all_zero_input_is_finite() -> None:
    assert numpy.isfinite(rms_dbfs(numpy.zeros(SAMPLE_RATE)))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        rms_dbfs(numpy.array([]))
    with pytest.raises(ValueError):
        leq_db([])


def test_leq_of_equal_levels_is_that_level() -> None:
    assert leq_db([60.0, 60.0, 60.0]) == pytest.approx(60.0)


def test_leq_is_energy_average_not_arithmetic() -> None:
    assert leq_db([60.0, 70.0]) == pytest.approx(67.4, abs=0.05)


def test_a_weighting_gain_at_1khz_is_zero() -> None:
    assert a_weighting_gain_db(1000.0) == pytest.approx(0.0, abs=0.5)


def test_a_weighting_gain_at_100hz() -> None:
    assert a_weighting_gain_db(100.0) == pytest.approx(-19.1, abs=0.5)


def test_block_filtering_matches_single_pass() -> None:
    signal = numpy.random.default_rng(0).standard_normal(SAMPLE_RATE)
    single_pass = AWeightingFilter(SAMPLE_RATE).process(signal)
    chunked_filter = AWeightingFilter(SAMPLE_RATE)
    # 일부러 고르지 않은 크기로 나눠서 경계 처리를 확인한다.
    chunks = numpy.array_split(signal, [1000, 1001, 7777, 30000])
    chunked = numpy.concatenate([chunked_filter.process(chunk) for chunk in chunks])
    numpy.testing.assert_allclose(chunked, single_pass, rtol=1e-9, atol=1e-12)


def test_level_meter_emits_eight_blocks_per_second() -> None:
    meter = LevelMeter(SAMPLE_RATE)
    assert len(meter.push(sine(1000.0, amplitude=0.1))) == 8


def test_level_meter_carries_partial_block() -> None:
    meter = LevelMeter(SAMPLE_RATE)
    block_length = round(config.FAST_BLOCK_SEC * SAMPLE_RATE)
    assert meter.push(numpy.zeros(block_length - 1)) == []
    assert len(meter.push(numpy.zeros(1))) == 1


def test_level_meter_steady_sine_level() -> None:
    meter = LevelMeter(SAMPLE_RATE)
    meter.push(sine(1000.0, amplitude=0.1, duration_sec=SETTLE_SEC))
    levels = meter.push(sine(1000.0, amplitude=0.1))
    # 진폭 0.1 사인 = -23.01 dBFS, 1kHz에서 A특성 게인 ≈ 0 dB.
    assert levels == pytest.approx([-23.01] * 8, abs=0.5)


def test_estimated_dba_uses_default_offset_for_unknown_mic() -> None:
    assert to_estimated_dba(-40.0, "unknown room") == pytest.approx(
        -40.0 + config.DEFAULT_CALIBRATION_OFFSET_DB
    )
