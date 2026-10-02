"""여러 마이크의 같은 시점 결과를 Frame 하나로 통합한다 (순수 로직)."""

from models import Frame, MicMeasurement


def fuse_measurements(
    timestamp: float, measurements: dict[str, MicMeasurement]
) -> Frame:
    """leq_db가 가장 큰 마이크를 대표로 삼는다. 위치·레벨·카테고리 확률 모두 대표 마이크 값."""
    if not measurements:
        raise ValueError("통합할 마이크 측정값이 없습니다")
    representative_name = max(measurements, key=lambda name: measurements[name].leq_db)
    representative = measurements[representative_name]
    return Frame(
        timestamp=timestamp,
        mic_name=representative_name,
        leq_db=representative.leq_db,
        lmax_db=representative.lmax_db,
        category_probs=representative.category_probs,
        top_label=representative.top_label,
    )
