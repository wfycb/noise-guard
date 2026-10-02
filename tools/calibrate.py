"""소음계 보정 도구 (뼈대만 있음, 미구현).

목적: 마이크별 dBFS(A) → dB(A) 오프셋을 실측해 calibration.json에 저장한다.
지금 config.DEFAULT_CALIBRATION_OFFSET_DB(100 dB)는 임시값이며, 이 도구로 대체한다.

의도한 사용 흐름 (방 하나, 마이크 하나씩):
    1. 마이크와 소음계를 같은 위치·방향에 나란히 둔다.
    2. 스피커로 핑크노이즈를 재생한다 (재생은 외부 플레이어로 해도 된다).
    3. 이 도구가 CALIBRATION_MEASURE_SEC 동안 마이크 dBFS(A) Leq를 측정한다.
    4. 같은 시간 동안 소음계(A특성, Fast)로 읽은 평균값을 사용자가 입력한다.
    5. offset_db = meter_dba - measured_dbfs_a 를 calibration.json에 방 이름으로 저장한다.

    .venv\\Scripts\\python -m tools.calibrate --mic 거실 [--device 18]

calibration.json 형식 (version 1):
    {
      "version": 1,
      "mics": {
        "거실": {
          "device": 18,                    # 측정에 쓴 장치 인덱스 또는 이름
          "offset_db": 96.4,               # meter_dba - measured_dbfs_a
          "measured_dbfs_a": -31.2,        # 측정 구간의 마이크 Leq, dBFS(A)
          "meter_dba": 65.2,               # 소음계 A특성·Fast 평균 입력값, dB(A)
          "duration_sec": 10.0,
          "reference": "pink noise",
          "measured_at": "2026-10-02T21:00:00+09:00"
        }
      }
    }
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import config

CALIBRATION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MicCalibration:
    device: int | str
    offset_db: float
    measured_dbfs_a: float
    meter_dba: float
    duration_sec: float
    reference: str
    measured_at: str  # ISO 8601, Asia/Seoul


def measure_dbfs_a(device: int | str, duration_sec: float) -> float:
    """마이크로 duration_sec 동안 녹음해 dBFS(A) Leq를 반환한다 (side effect: 장치 접근).

    TODO: capture.MicSource + level.LevelMeter로 블록 레벨을 모아 level.leq_db로 계산.
    """
    raise NotImplementedError("보정 측정은 아직 구현되지 않았습니다")


def ask_meter_reading() -> float:
    """소음계 A특성·Fast 평균값(dB(A))을 사용자에게 입력받는다 (side effect: 콘솔 입력).

    TODO: 숫자가 아니면 다시 묻기.
    """
    raise NotImplementedError("소음계 입력은 아직 구현되지 않았습니다")


def save_calibration(path: Path, mic_name: str, calibration: MicCalibration) -> None:
    """calibration.json에 방 하나의 결과를 추가/갱신한다 (side effect: 파일 쓰기).

    TODO: 기존 파일을 읽어 mics[mic_name]만 바꾸고 version을 확인한 뒤 저장.
    """
    raise NotImplementedError("보정 결과 저장은 아직 구현되지 않았습니다")


def main() -> None:
    parser = argparse.ArgumentParser(description="소음계 보정 (미구현 뼈대)")
    parser.add_argument("--mic", required=True, help="방 이름 (config.MIC_DEVICES 키)")
    parser.add_argument("--device", help="장치 인덱스/이름 (생략 시 MIC_DEVICES 값)")
    parser.add_argument("--output", type=Path, default=Path(config.CALIBRATION_FILE))
    parser.parse_args()
    raise SystemExit(
        "tools/calibrate.py는 아직 뼈대만 있습니다. 사용 흐름과 calibration.json 형식은 "
        "이 파일의 docstring을 참고하세요."
    )


if __name__ == "__main__":
    main()
