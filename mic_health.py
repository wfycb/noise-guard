"""마이크 빠짐 추적 (순수 로직). tick마다 레벨·분류를 낸 마이크 목록을 받아 상태 변화를 알려준다."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MicStatusChange:
    """빠진 마이크 목록이 바뀐 시점의 정보. 바뀌지 않은 tick에는 만들지 않는다."""

    timestamp: float
    # 지금 "이상"으로 보는 방 (warn_ticks 이상 연속으로 빠짐)
    missing_mics: tuple[str, ...]
    newly_missing: tuple[str, ...]
    recovered: tuple[str, ...]
    # 새로 빠짐: 연속으로 빠진 시간, 복구: 빠져 있던 시간
    missing_seconds: dict[str, float]
    active_mics: int
    total_mics: int


@dataclass
class _MicPresence:
    streak: int = 0  # 연속으로 빠진 tick 수
    total_missing_ticks: int = 0
    warned: bool = False


@dataclass
class MicPresenceTracker:
    """연속 warn_ticks번 빠지면 "이상", 다시 들어오면 "복구"로 본다."""

    mic_names: list[str]
    warn_ticks: int
    hop_sec: float
    _presence: dict[str, _MicPresence] = field(init=False)
    last_active_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._presence = {name: _MicPresence() for name in self.mic_names}

    @property
    def total_mics(self) -> int:
        return len(self.mic_names)

    def missing_tick_counts(self) -> dict[str, int]:
        return {
            name: state.total_missing_ticks for name, state in self._presence.items()
        }

    def update(self, timestamp: float, present: set[str]) -> MicStatusChange | None:
        """tick 하나를 반영한다. 빠진 마이크 목록이 바뀌었을 때만 변화를 반환한다."""
        newly_missing = []
        recovered = []
        missing_seconds = {}
        for name, state in self._presence.items():
            if name in present:
                if state.warned:
                    recovered.append(name)
                    missing_seconds[name] = state.streak * self.hop_sec
                state.streak = 0
                state.warned = False
                continue
            state.streak += 1
            state.total_missing_ticks += 1
            if not state.warned and state.streak >= self.warn_ticks:
                state.warned = True
                newly_missing.append(name)
                missing_seconds[name] = state.streak * self.hop_sec
        self.last_active_count = sum(name in present for name in self.mic_names)
        if not newly_missing and not recovered:
            return None
        return MicStatusChange(
            timestamp=timestamp,
            missing_mics=tuple(
                name for name, state in self._presence.items() if state.warned
            ),
            newly_missing=tuple(newly_missing),
            recovered=tuple(recovered),
            missing_seconds=missing_seconds,
            active_mics=self.last_active_count,
            total_mics=self.total_mics,
        )
