import pytest

from fusion import fuse_measurements
from models import Category, MicMeasurement

TIMESTAMP = 1_000_000.0


def make_measurement(leq_db: float, impact: float = 0.0) -> MicMeasurement:
    return MicMeasurement(
        leq_db=leq_db,
        lmax_db=leq_db + 3.0,
        category_probs={
            Category.IMPACT: impact,
            Category.AIRBORNE: 0.0,
            Category.EXCLUDED: 0.0,
            Category.OTHER: 0.0,
        },
        top_label=f"label-{leq_db}",
    )


def test_loudest_mic_becomes_representative() -> None:
    frame = fuse_measurements(
        TIMESTAMP,
        {
            "거실": make_measurement(50.0, impact=0.1),
            "안방": make_measurement(62.0, impact=0.8),
            "주방": make_measurement(55.0, impact=0.3),
        },
    )
    assert frame.mic_name == "안방"
    assert frame.leq_db == 62.0
    assert frame.lmax_db == 65.0
    assert frame.category_probs[Category.IMPACT] == 0.8
    assert frame.top_label == "label-62.0"
    assert frame.timestamp == TIMESTAMP


def test_single_mic_works() -> None:
    frame = fuse_measurements(TIMESTAMP, {"거실": make_measurement(40.0)})
    assert frame.mic_name == "거실"
    assert frame.leq_db == 40.0


def test_no_measurements_raises() -> None:
    with pytest.raises(ValueError):
        fuse_measurements(TIMESTAMP, {})
