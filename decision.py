"""경고 판단 로직 (순수 로직). 시간은 frame.timestamp만 쓰고 내부에서 시계를 읽지 않는다."""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy

import config
from models import Alert, Category, Frame, TimePeriod

JUDGED_CATEGORIES: tuple[Category, ...] = (Category.IMPACT, Category.AIRBORNE)
LOCAL_TIMEZONE = timezone(timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS))
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

CAUTION = "caution"
WARNING = "warning"


def judge_category(
    category_probs: dict[Category, float], thresholds: dict[Category, float]
) -> Category | None:
    """IMPACT/AIRBORNE 중 각자의 임계값을 넘은 것 중 확률이 더 높은 쪽. 없으면 None(스킵).

    EXCLUDED 확률은 보지 않는다. 물소리와 발소리가 겹쳐도 IMPACT가 임계값을 넘으면
    IMPACT로 판정해야 하기 때문이다 (EXCLUDED는 "단독일 때만 스킵").
    둘 다 넘으면 임계값 대비 여유가 아니라 원래 확률로 비교한다 (사양 9.2).
    """
    candidates = [
        category
        for category in JUDGED_CATEGORIES
        if category_probs.get(category, 0.0) >= thresholds[category]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda category: category_probs[category])


def time_period(timestamp: float) -> TimePeriod:
    """로컬(Asia/Seoul) 시각 기준 주간(06–22시)/야간(22–06시)."""
    hour = datetime.fromtimestamp(timestamp, LOCAL_TIMEZONE).hour
    if config.DAY_START_HOUR <= hour < config.NIGHT_START_HOUR:
        return TimePeriod.DAY
    return TimePeriod.NIGHT


def windowed_leq_db(
    levels_db: list[float], window_sec: float, frame_sec: float, floor_db: float
) -> float:
    """창 전체 길이 기준 Leq (사양 9.5 (a)). 프레임이 없는 구간은 floor_db로 채운다."""
    frame_energy = float(
        numpy.sum(10.0 ** (numpy.asarray(levels_db, dtype=numpy.float64) / 10.0))
    )
    floor_duration = max(window_sec - len(levels_db) * frame_sec, 0.0)
    total_energy = frame_energy * frame_sec + floor_duration * 10.0 ** (floor_db / 10.0)
    return float(10.0 * numpy.log10(total_energy / window_sec))


def format_duration(seconds: float) -> str:
    """메시지용 시간창 표기: 3600 → "1시간", 300 → "5분", 20 → "20초"."""
    if seconds >= SECONDS_PER_HOUR and seconds % SECONDS_PER_HOUR == 0:
        return f"{int(seconds // SECONDS_PER_HOUR)}시간"
    if seconds >= SECONDS_PER_MINUTE and seconds % SECONDS_PER_MINUTE == 0:
        return f"{int(seconds // SECONDS_PER_MINUTE)}분"
    return f"{seconds:g}초"


def format_room_counts(room_counts: dict[str, int]) -> str:
    """{"거실": 2, "안방": 1} → "거실 2회, 안방 1회" (입력 순서 유지)."""
    return ", ".join(f"{room} {count}회" for room, count in room_counts.items())


@dataclass(frozen=True)
class DecisionConfig:
    class_prob_threshold: dict[Category, float]
    impact_leq_limit_db: dict[TimePeriod, float]
    impact_lmax_limit_db: dict[TimePeriod, float]
    airborne_leq_limit_db: dict[TimePeriod, float]
    lmax_count_to_warn: int
    lmax_window_sec: float
    impact_leq_window_sec: float
    airborne_leq_window_sec: float
    merge_gap_sec: float
    cooldown_sec: float
    frame_sec: float
    leq_floor_db: float

    @classmethod
    def from_config(cls, demo_mode: bool) -> "DecisionConfig":
        """config.py 상수로 만든다. demo_mode면 시간창과 쿨다운만 시연용 값으로 바꾼다."""
        return cls(
            class_prob_threshold=dict(config.CLASS_PROB_THRESHOLD),
            impact_leq_limit_db={
                TimePeriod.DAY: config.IMPACT_LEQ_LIMIT_DAY_DB,
                TimePeriod.NIGHT: config.IMPACT_LEQ_LIMIT_NIGHT_DB,
            },
            impact_lmax_limit_db={
                TimePeriod.DAY: config.IMPACT_LMAX_LIMIT_DAY_DB,
                TimePeriod.NIGHT: config.IMPACT_LMAX_LIMIT_NIGHT_DB,
            },
            airborne_leq_limit_db={
                TimePeriod.DAY: config.AIRBORNE_LEQ_LIMIT_DAY_DB,
                TimePeriod.NIGHT: config.AIRBORNE_LEQ_LIMIT_NIGHT_DB,
            },
            lmax_count_to_warn=config.LMAX_COUNT_TO_WARN,
            lmax_window_sec=(
                config.DEMO_LMAX_WINDOW_SEC if demo_mode else config.LMAX_WINDOW_SEC
            ),
            impact_leq_window_sec=(
                config.DEMO_IMPACT_LEQ_WINDOW_SEC
                if demo_mode
                else config.IMPACT_LEQ_WINDOW_SEC
            ),
            airborne_leq_window_sec=(
                config.DEMO_AIRBORNE_LEQ_WINDOW_SEC
                if demo_mode
                else config.AIRBORNE_LEQ_WINDOW_SEC
            ),
            merge_gap_sec=config.MERGE_GAP_SEC,
            cooldown_sec=(
                config.DEMO_COOLDOWN_SEC if demo_mode else config.COOLDOWN_SEC
            ),
            frame_sec=config.FRAME_SEC,
            leq_floor_db=config.LEQ_FLOOR_DB,
        )


@dataclass
class ImpactEvent:
    """MERGE_GAP_SEC 이내로 이어진 IMPACT 프레임 묶음."""

    last_timestamp: float
    lmax_db: float
    counted: bool = False


class NoiseDecisionEngine:
    """프레임을 받아 R1~R4 규칙으로 알림을 낸다.

    - R1: IMPACT 이벤트 Lmax가 처음 기준을 넘는 프레임에서 caution 1회, 카운트 +1.
    - R2: R1 카운트가 늘어날 때만 평가. 매 프레임 평가하면 카운트가 남아 있는 동안
      쿨다운마다 같은 경고가 반복되기 때문이다.
    - R3/R4: 해당 카테고리 프레임이 들어올 때만 평가. 소음이 멈춘 뒤 창에 남은 Leq로
      경고가 반복되지 않게 하기 위해서다.
    """

    def __init__(self, decision_config: DecisionConfig) -> None:
        self._config = decision_config
        self._frames_by_category: dict[Category, deque[tuple[float, float]]] = {
            category: deque() for category in JUDGED_CATEGORIES
        }
        # (시각, 방 이름). 카운트는 집 전체 기준이고, 방 이름은 R2 메시지의 내역용이다.
        self._lmax_exceeds: deque[tuple[float, str]] = deque()
        self._current_impact_event: ImpactEvent | None = None
        self._last_warning_timestamp: dict[str, float] = {}
        self._longest_window_sec = max(
            decision_config.lmax_window_sec,
            decision_config.impact_leq_window_sec,
            decision_config.airborne_leq_window_sec,
        )

    @property
    def stored_frame_count(self) -> int:
        return sum(len(frames) for frames in self._frames_by_category.values())

    @property
    def stored_lmax_exceed_count(self) -> int:
        return len(self._lmax_exceeds)

    def update(self, frame: Frame) -> list[Alert]:
        """프레임 하나를 반영하고 이번에 발생한 알림을 반환한다."""
        self._prune(frame.timestamp)
        category = judge_category(
            frame.category_probs, self._config.class_prob_threshold
        )
        if category is None:
            return []
        self._frames_by_category[category].append((frame.timestamp, frame.leq_db))
        period = time_period(frame.timestamp)
        if category == Category.IMPACT:
            return self._update_impact(frame, period)
        return self._check_leq_rule(
            "R4",
            Category.AIRBORNE,
            frame,
            period,
            self._config.airborne_leq_window_sec,
            self._config.airborne_leq_limit_db[period],
        )

    def _update_impact(self, frame: Frame, period: TimePeriod) -> list[Alert]:
        alerts = self._update_impact_event(frame, period)
        alerts += self._check_leq_rule(
            "R3",
            Category.IMPACT,
            frame,
            period,
            self._config.impact_leq_window_sec,
            self._config.impact_leq_limit_db[period],
        )
        return alerts

    def _update_impact_event(self, frame: Frame, period: TimePeriod) -> list[Alert]:
        """R1(caution)과 R2(warning). 같은 이벤트에서는 caution을 한 번만 낸다."""
        event = self._current_impact_event
        if (
            event is None
            or frame.timestamp - event.last_timestamp > self._config.merge_gap_sec
        ):
            event = ImpactEvent(last_timestamp=frame.timestamp, lmax_db=frame.lmax_db)
            self._current_impact_event = event
        else:
            event.last_timestamp = frame.timestamp
            event.lmax_db = max(event.lmax_db, frame.lmax_db)

        lmax_limit = self._config.impact_lmax_limit_db[period]
        if event.counted or not event.lmax_db >= lmax_limit:
            return []
        event.counted = True
        self._lmax_exceeds.append((frame.timestamp, frame.mic_name))
        count = len(self._lmax_exceeds)
        room_counts = dict(
            Counter(mic_name for _, mic_name in self._lmax_exceeds).most_common()
        )
        window_text = format_duration(self._config.lmax_window_sec)
        alerts = [
            Alert(
                level=CAUTION,
                rule="R1",
                category=Category.IMPACT,
                mic_name=frame.mic_name,
                peak_db=event.lmax_db,
                count=count,
                timestamp=frame.timestamp,
                message=(
                    f"[주의] {frame.mic_name}에서 충격소음({frame.top_label}) 감지 — "
                    f"최대 {event.lmax_db:.1f} dB(A), 최근 {window_text} {count}회"
                ),
                room_counts=room_counts,
            )
        ]
        if count >= self._config.lmax_count_to_warn and self._cooldown_passed(
            "R2", frame.timestamp
        ):
            self._last_warning_timestamp["R2"] = frame.timestamp
            alerts.append(
                Alert(
                    level=WARNING,
                    rule="R2",
                    category=Category.IMPACT,
                    mic_name=frame.mic_name,
                    peak_db=event.lmax_db,
                    count=count,
                    timestamp=frame.timestamp,
                    message=(
                        f"[경고] {frame.mic_name} 충격소음 반복 — 최근 {window_text} "
                        f"{count}회 기준 초과 ({format_room_counts(room_counts)} / "
                        f"{period.value} 기준 {lmax_limit:.0f} dB(A))"
                    ),
                    room_counts=room_counts,
                )
            )
        return alerts

    def _check_leq_rule(
        self,
        rule: str,
        category: Category,
        frame: Frame,
        period: TimePeriod,
        window_sec: float,
        limit_db: float,
    ) -> list[Alert]:
        """창 안의 해당 카테고리 프레임으로 Leq를 계산해 기준 이상이면 warning (R3/R4)."""
        window_start = frame.timestamp - window_sec
        levels = [
            leq
            for timestamp, leq in self._frames_by_category[category]
            if timestamp > window_start
        ]
        if not levels:
            return []
        leq = windowed_leq_db(
            levels, window_sec, self._config.frame_sec, self._config.leq_floor_db
        )
        if not leq >= limit_db or not self._cooldown_passed(rule, frame.timestamp):
            return []
        self._last_warning_timestamp[rule] = frame.timestamp
        category_name = "충격소음" if category == Category.IMPACT else "공기전달소음"
        return [
            Alert(
                level=WARNING,
                rule=rule,
                category=category,
                mic_name=frame.mic_name,
                peak_db=leq,
                count=1,
                timestamp=frame.timestamp,
                message=(
                    f"[경고] {frame.mic_name} {category_name} 지속 — "
                    f"최근 {format_duration(window_sec)} Leq {leq:.1f} dB(A) "
                    f"({period.value} 기준 {limit_db:.0f} dB(A))"
                ),
            )
        ]

    def _cooldown_passed(self, rule: str, timestamp: float) -> bool:
        last_timestamp = self._last_warning_timestamp.get(rule)
        return (
            last_timestamp is None
            or timestamp - last_timestamp >= self._config.cooldown_sec
        )

    def _prune(self, now: float) -> None:
        """가장 긴 시간창 밖의 프레임과 Lmax 창 밖의 카운트를 지워 메모리가 늘지 않게 한다."""
        frame_cutoff = now - self._longest_window_sec
        for frames in self._frames_by_category.values():
            while frames and frames[0][0] <= frame_cutoff:
                frames.popleft()
        lmax_cutoff = now - self._config.lmax_window_sec
        while self._lmax_exceeds and self._lmax_exceeds[0][0] <= lmax_cutoff:
            self._lmax_exceeds.popleft()
