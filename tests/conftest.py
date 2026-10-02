"""테스트 세션 공통 설정."""

from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"


@pytest.fixture(scope="session", autouse=True)
def no_calibration_file_written_to_project_root() -> Iterator[None]:
    """테스트가 프로젝트 루트에 calibration.json을 만들면 실패시킨다.

    루트의 calibration.json은 실제 소음계로 잰 값만 있어야 한다. 테스트·예시는 tmp_path나 data/ 아래에 쓴다.
    """
    existed_before = ROOT_CALIBRATION_FILE.exists()
    yield
    if not existed_before and ROOT_CALIBRATION_FILE.exists():
        pytest.fail(f"테스트가 {ROOT_CALIBRATION_FILE}을 만들었습니다")
