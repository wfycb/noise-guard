from itertools import combinations
from pathlib import Path

import numpy
import pytest

from label_map import (
    AIRBORNE_EXTENDED_LABELS,
    AIRBORNE_LEGAL_LABELS,
    EXCLUDED_LABELS,
    IMPACT_LABELS,
    build_category_indices,
    category_for_label,
    category_probabilities,
)
from models import Category

LABELS_DUMP_PATH = Path(__file__).resolve().parent.parent / "labels_dump.txt"

LABEL_SETS = {
    "IMPACT": IMPACT_LABELS,
    "AIRBORNE_LEGAL": AIRBORNE_LEGAL_LABELS,
    "AIRBORNE_EXTENDED": AIRBORNE_EXTENDED_LABELS,
    "EXCLUDED": EXCLUDED_LABELS,
}


def load_dumped_labels() -> list[str]:
    lines = LABELS_DUMP_PATH.read_text(encoding="utf-8").splitlines()
    return [line.split("\t", 1)[1] for line in lines]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Walk, footsteps", Category.IMPACT),
        ("Thump, thud", Category.IMPACT),
        ("Smash, crash", Category.IMPACT),
        ("Television", Category.AIRBORNE),
        ("Piano", Category.AIRBORNE),
        ("Toilet flush", Category.EXCLUDED),
        ("Stream", Category.EXCLUDED),
        ("Clapping", Category.OTHER),
        ("Applause", Category.OTHER),
        ("Silence", Category.OTHER),
    ],
)
def test_known_labels_map_to_expected_category(label: str, expected: Category) -> None:
    assert category_for_label(label, "extended") == expected
    assert category_for_label(label, "legal") == expected


def test_speech_mapping_depends_on_scope() -> None:
    assert category_for_label("Speech", "extended") == Category.AIRBORNE
    assert category_for_label("Speech", "legal") == Category.OTHER


def test_dog_and_vacuum_are_extended_only() -> None:
    for label in ("Bark", "Vacuum cleaner"):
        assert category_for_label(label, "extended") == Category.AIRBORNE
        assert category_for_label(label, "legal") == Category.OTHER


def test_unknown_label_maps_to_other() -> None:
    assert category_for_label("Not a real AudioSet label", "extended") == Category.OTHER


def test_unknown_scope_raises() -> None:
    with pytest.raises(ValueError):
        category_for_label("Speech", "everything")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("first_name", "second_name"), list(combinations(LABEL_SETS, 2))
)
def test_label_sets_do_not_overlap(first_name: str, second_name: str) -> None:
    overlap = LABEL_SETS[first_name] & LABEL_SETS[second_name]
    assert not overlap, f"{first_name}와 {second_name}에 중복: {sorted(overlap)}"


def test_every_mapped_label_exists_in_model_labels() -> None:
    # 오타가 있으면 조용히 OTHER로 빠지므로 실제 id2label 덤프와 대조한다.
    dumped_labels = set(load_dumped_labels())
    for name, label_set in LABEL_SETS.items():
        missing = label_set - dumped_labels
        assert not missing, f"{name}에 덤프에 없는 라벨: {sorted(missing)}"


def test_category_probabilities_take_max_within_category() -> None:
    labels = ["Walk, footsteps", "Bang", "Speech", "Water", "Silence"]
    probabilities = numpy.array([0.2, 0.7, 0.4, 0.9, 0.1])
    category_indices = build_category_indices(labels, "extended")
    result = category_probabilities(probabilities, category_indices)
    assert result[Category.IMPACT] == pytest.approx(0.7)
    assert result[Category.AIRBORNE] == pytest.approx(0.4)
    assert result[Category.EXCLUDED] == pytest.approx(0.9)
    assert result[Category.OTHER] == pytest.approx(0.1)


def test_category_without_labels_has_zero_probability() -> None:
    labels = ["Speech", "Silence"]
    category_indices = build_category_indices(labels, "legal")
    result = category_probabilities(numpy.array([0.8, 0.1]), category_indices)
    assert result[Category.AIRBORNE] == 0.0
    assert result[Category.OTHER] == pytest.approx(0.8)
