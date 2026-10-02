"""소음계 보정 결과(calibration.json)의 계산·검증·저장·로딩.

계산과 검증은 순수 함수다. 파일 입출력과 config 변경은 함수 이름과 docstring에 표시한다.

calibration.json 형식 (방 이름 → 결과):
    {"거실": {"offset_db": 93.4, "meter_leq": 62.0, "program_leq": -31.4,
             "levels": [{"meter_leq": 62.0, "program_leq": -31.4, "offset_db": 93.4}, ...],
             "background_leq_db": 28.1, "sample_rate": 48000, "device": "18",
             "measured_at": "2026-10-10T14:00:00+09:00"}}
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import config

logger = logging.getLogger(__name__)


class CalibrationFileError(ValueError):
    """calibration.json을 읽을 수 없거나 형식이 틀렸다."""


@dataclass(frozen=True)
class LevelReading:
    """핑크노이즈 볼륨 한 단계의 측정값."""

    meter_leq: float  # 소음계 A특성 Leq, dB(A)
    program_leq: float  # 같은 시간의 프로그램 Leq, dBFS(A)
    offset_db: float


@dataclass(frozen=True)
class MicCalibration:
    offset_db: float  # 단계별 오프셋의 평균
    meter_leq: float  # 단계별 평균 (기록용)
    program_leq: float
    levels: tuple[LevelReading, ...]
    background_leq_db: float  # 보정 후 배경 소음 Leq, dB(A)
    sample_rate: int
    device: str
    measured_at: str  # ISO 8601, Asia/Seoul


def compute_offset(meter_leq: float, program_leq: float) -> float:
    """오프셋 = 소음계 Leq − 프로그램 Leq. 이후 추정 dB(A) = dBFS(A) + 오프셋."""
    return meter_leq - program_leq


def make_reading(meter_leq: float, program_leq: float) -> LevelReading:
    return LevelReading(meter_leq, program_leq, compute_offset(meter_leq, program_leq))


def offset_spread_db(readings: list[LevelReading]) -> float:
    """단계별 오프셋의 최대 − 최소. 마이크 AGC가 켜져 있으면 커진다."""
    offsets = [reading.offset_db for reading in readings]
    return max(offsets) - min(offsets)


def spread_warning(readings: list[LevelReading], max_spread_db: float) -> str | None:
    """단계별 오프셋 차이가 허용치를 넘으면 경고 문장, 아니면 None."""
    if len(readings) < 2:
        return None
    spread = offset_spread_db(readings)
    if spread <= max_spread_db:
        return None
    return (
        f"볼륨별 오프셋 차이 {spread:.1f} dB > {max_spread_db} dB: 마이크 AGC(자동 이득)가 "
        "켜져 있거나 측정이 불안정합니다. 레벨에 따라 dB가 틀어질 수 있습니다"
    )


def evaluate_background(
    background_leq_db: float,
    gate_db: float,
    min_limit_db: float,
    margin_db: float,
    near_limit_db: float,
    headroom_db: float,
) -> tuple[list[str], float]:
    """배경 소음으로 게이트·기준을 점검한다. (경고 목록, 권장 게이트)를 반환한다 (순수 함수).

    권장 게이트 = min(배경 + headroom, 최저 기준 − 마진). 게이트 값은 자동으로 바꾸지 않는다.
    """
    warnings = []
    if background_leq_db >= gate_db:
        warnings.append(
            f"게이트 {gate_db} dB(A)가 배경 소음 {background_leq_db:.1f} dB(A)보다 낮아 "
            "건너뛰기가 일어나지 않음"
        )
    if background_leq_db >= min_limit_db - near_limit_db:
        warnings.append(
            f"배경 소음 {background_leq_db:.1f} dB(A)가 최저 판단 기준 {min_limit_db} dB(A)에 "
            f"가까움({near_limit_db} dB 이내), 오탐 위험"
        )
    recommended_gate = min(background_leq_db + headroom_db, min_limit_db - margin_db)
    return warnings, recommended_gate


def build_calibration(
    readings: list[LevelReading],
    background_dbfs_a: float,
    sample_rate: int,
    device: str,
    measured_at: str,
) -> MicCalibration:
    """단계별 측정값과 배경 dBFS(A)로 결과를 만든다. 배경은 확정된 오프셋으로 dB(A)로 바꾼다."""
    offset = fmean(reading.offset_db for reading in readings)
    return MicCalibration(
        offset_db=offset,
        meter_leq=fmean(reading.meter_leq for reading in readings),
        program_leq=fmean(reading.program_leq for reading in readings),
        levels=tuple(readings),
        background_leq_db=background_dbfs_a + offset,
        sample_rate=sample_rate,
        device=device,
        measured_at=measured_at,
    )


def calibration_to_json(calibration: MicCalibration) -> dict[str, Any]:
    payload = asdict(calibration)
    payload["levels"] = [asdict(reading) for reading in calibration.levels]
    return payload


def calibration_from_json(room: str, payload: Any) -> MicCalibration:
    """json 한 방의 값을 읽는다. 형식이 틀리면 방 이름을 넣어 CalibrationFileError."""
    try:
        if not isinstance(payload, dict):
            raise TypeError("객체가 아닙니다")
        levels = tuple(
            LevelReading(
                float(level["meter_leq"]),
                float(level["program_leq"]),
                float(level["offset_db"]),
            )
            for level in payload.get("levels", [])
        )
        return MicCalibration(
            offset_db=float(payload["offset_db"]),
            meter_leq=float(payload.get("meter_leq", float("nan"))),
            program_leq=float(payload.get("program_leq", float("nan"))),
            levels=levels,
            background_leq_db=float(payload.get("background_leq_db", float("nan"))),
            sample_rate=int(payload.get("sample_rate", 0)),
            device=str(payload.get("device", "")),
            measured_at=str(payload.get("measured_at", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationFileError(
            f"'{room}' 보정값 형식이 틀립니다 ({type(error).__name__}: {error}). "
            "offset_db(숫자)가 필요합니다"
        ) from error


def load_calibration(path: Path) -> dict[str, MicCalibration]:
    """calibration.json을 읽는다 (side effect: 파일 읽기). 파일이 없으면 빈 dict."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CalibrationFileError(
            f"{path}: JSON 형식 오류 ({error.lineno}번째 줄, {error.colno}번째 글자: "
            f"{error.msg}). 파일을 고치거나 지우고 다시 보정하세요"
        ) from error
    except UnicodeDecodeError as error:
        raise CalibrationFileError(f"{path}: UTF-8로 읽을 수 없습니다") from error
    if not isinstance(payload, dict):
        raise CalibrationFileError(f"{path}: 최상위가 방 이름 → 보정값 객체여야 합니다")
    return {room: calibration_from_json(room, value) for room, value in payload.items()}


def save_mic_calibration(path: Path, room: str, calibration: MicCalibration) -> None:
    """방 하나의 결과를 추가·갱신해 저장한다 (side effect: 파일 읽기·쓰기).

    기존 파일이 깨져 있으면 덮어쓰지 않고 CalibrationFileError를 낸다(다른 방의 결과를 잃지 않게).
    """
    existing = load_calibration(path)
    payload = {name: calibration_to_json(value) for name, value in existing.items()}
    payload[room] = calibration_to_json(calibration)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def apply_calibration(calibrations: dict[str, MicCalibration]) -> None:
    """방별 오프셋을 config.CALIBRATION_OFFSET_DB에 넣는다 (side effect: config 변경).

    level.to_estimated_dba가 이 dict를 읽는다. 없는 방은 DEFAULT_CALIBRATION_OFFSET_DB(임시)를 쓴다.
    """
    config.CALIBRATION_OFFSET_DB.clear()
    config.CALIBRATION_OFFSET_DB.update(
        {room: value.offset_db for room, value in calibrations.items()}
    )


def is_calibrated(room: str) -> bool:
    return room in config.CALIBRATION_OFFSET_DB


def warn_uncalibrated(rooms: list[str]) -> list[str]:
    """보정되지 않은 방을 경고 로그로 남기고 목록을 반환한다 (side effect: 로그)."""
    missing = [room for room in rooms if not is_calibrated(room)]
    if missing:
        logger.warning(
            "보정 안 됨: %s — 임시 오프셋 +%.0f dB 사용, dB 값은 실제 SPL이 아님",
            ", ".join(missing),
            config.DEFAULT_CALIBRATION_OFFSET_DB,
        )
    return missing
