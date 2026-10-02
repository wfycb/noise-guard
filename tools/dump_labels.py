"""CED의 id2label 전체를 파일로 저장한다 (side effect: 모델 설정 다운로드, 파일 쓰기).

실행 (프로젝트 루트에서):
    .venv\\Scripts\\python -m tools.dump_labels [--output labels_dump.txt]
"""

import argparse
from pathlib import Path

from transformers import AutoConfig

import config

DEFAULT_OUTPUT_PATH = Path("labels_dump.txt")


def load_id2label() -> dict[int, str]:
    """가중치 없이 모델 설정만 받아 id2label을 반환한다 (side effect: 다운로드)."""
    model_config = AutoConfig.from_pretrained(
        config.CED_MODEL_NAME,
        revision=config.CED_MODEL_REVISION,
        trust_remote_code=True,
    )
    return {int(index): label for index, label in model_config.id2label.items()}


def write_labels(id2label: dict[int, str], output_path: Path) -> None:
    """한 줄에 "인덱스<TAB>라벨" 형식으로 저장한다 (side effect: 파일 쓰기)."""
    lines = [f"{index}\t{id2label[index]}" for index in sorted(id2label)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CED id2label 덤프")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    id2label = load_id2label()
    write_labels(id2label, arguments.output)
    print(f"{len(id2label)}개 라벨 → {arguments.output}")
