"""공용 데이터 구조."""

from dataclasses import dataclass, field
from enum import Enum


class Category(Enum):
    IMPACT = "impact"  # 직접충격소음: 발소리, 쿵, 문 쾅 등
    AIRBORNE = "airborne"  # 공기전달소음: TV, 음악 (extended 모드에서는 말소리 등 포함)
    EXCLUDED = "excluded"  # 법적 제외: 급배수(물소리, 샤워, 변기 등)
    OTHER = "other"  # 그 외 / 무음


class TimePeriod(Enum):
    DAY = "주간"
    NIGHT = "야간"


@dataclass(frozen=True)
class MicMeasurement:
    """마이크 하나의 1초 측정·분류 결과. fusion이 이것들을 Frame 하나로 합친다."""

    leq_db: float
    lmax_db: float
    category_probs: dict[Category, float]
    top_label: str


@dataclass(frozen=True)
class Frame:
    timestamp: float  # 프레임 끝 시각 (epoch sec)
    mic_name: str  # 대표 마이크(레벨 최대)의 방 이름
    leq_db: float  # 1초 Leq, dB(A) 추정값
    lmax_db: float  # 1초 내 Fast(125ms) 블록 최대값
    category_probs: dict[Category, float]  # 카테고리별 최대 sigmoid 확률
    top_label: str  # 최고 확률 AudioSet 라벨


@dataclass(frozen=True)
class Alert:
    level: str  # "caution" | "warning"
    rule: str  # "R1" ~ "R4"
    category: Category
    mic_name: str
    peak_db: float
    count: int  # 해당 시간창 내 횟수 (R1/R2), Leq 규칙이면 1
    timestamp: float
    message: str  # 사람이 읽을 한국어 메시지
    # R1/R2: 시간창 내 Lmax 초과 횟수의 방별 내역 (합계 = count). Leq 규칙은 비어 있다.
    # 메시지 문자열을 다시 파싱하지 않고 OLED 표시·CSV·테스트에서 쓰기 위해 구조화해 둔다.
    room_counts: dict[str, int] = field(default_factory=dict)
