"""wav 파일/폴더를 CED로 분류해 카테고리 판정을 검증한다 (side effect: 파일 읽기, 콘솔 출력).

내장 마이크가 말소리 외 소리를 걸러내므로, 충격음·물소리 등은 이 도구로 파일 기반 검증한다.

실행 (프로젝트 루트에서):
    # 파일 하나: 창마다 카테고리 확률과 top-5 출력
    .venv\\Scripts\\python -m tools.classify_file some.wav
    # 폴더 + ESC-50 메타데이터: 클래스별 판정 분포 요약
    .venv\\Scripts\\python -m tools.classify_file data/esc50/audio \\
        --meta data/esc50/meta/esc50.csv --classes footsteps,dog
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy

import config
from capture import read_wav_mono
from classifier import CedClassifier, ClassificationResult, resample_for_classifier
from decision import judge_category
from models import Category

SKIP_COLUMN = "skip"
SUMMARY_CATEGORIES: tuple[Category, ...] = (
    Category.IMPACT,
    Category.AIRBORNE,
    Category.EXCLUDED,
)


@dataclass(frozen=True)
class FileResult:
    path: Path
    class_name: str
    window_results: list[ClassificationResult]


def split_windows(
    samples: numpy.ndarray, sample_rate: int, window_sec: float, hop_sec: float
) -> list[numpy.ndarray]:
    """window_sec 창을 hop_sec 간격으로 자른다. 창보다 짧으면 0으로 채워 창 1개로 만든다."""
    window_length = int(window_sec * sample_rate)
    hop_length = int(hop_sec * sample_rate)
    if len(samples) < window_length:
        padded = numpy.zeros(window_length, dtype=numpy.float32)
        padded[: len(samples)] = samples
        return [padded]
    starts = range(0, len(samples) - window_length + 1, hop_length)
    return [samples[start : start + window_length] for start in starts]


def classify_wav(classifier: CedClassifier, path: Path) -> list[ClassificationResult]:
    """파일 하나의 모든 창을 한 번의 forward로 분류한다 (side effect: 파일 읽기)."""
    samples, sample_rate = read_wav_mono(path)
    resampled = resample_for_classifier(samples, sample_rate)
    windows = split_windows(
        resampled,
        config.CLASSIFIER_SAMPLE_RATE,
        config.CLASSIFY_WINDOW_SEC,
        config.CLASSIFY_HOP_SEC,
    )
    probabilities = classifier.predict_probabilities(windows)
    return [classifier.build_result(row) for row in probabilities]


def load_class_names(meta_path: Path) -> dict[str, str]:
    """ESC-50 형식 CSV(filename, category 열)에서 파일명 → 클래스 이름 (side effect: 파일 읽기)."""
    with meta_path.open(encoding="utf-8", newline="") as meta_file:
        return {row["filename"]: row["category"] for row in csv.DictReader(meta_file)}


def collect_wav_files(
    folder: Path, class_names: dict[str, str] | None, wanted_classes: set[str] | None
) -> list[tuple[Path, str]]:
    """(경로, 클래스) 목록. 메타데이터가 없으면 상위 폴더 이름을 클래스로 쓴다."""
    collected = []
    for path in sorted(folder.rglob("*.wav")):
        if class_names is not None:
            class_name = class_names.get(path.name)
            if class_name is None:
                continue
        else:
            class_name = path.parent.name
        if wanted_classes is None or class_name in wanted_classes:
            collected.append((path, class_name))
    return collected


Thresholds = dict[Category, float]


def uniform_thresholds(value: float) -> Thresholds:
    """IMPACT와 AIRBORNE에 같은 임계값을 적용한다."""
    return {Category.IMPACT: value, Category.AIRBORNE: value}


def format_thresholds(thresholds: Thresholds) -> str:
    return (
        f"IMPACT {thresholds[Category.IMPACT]:.2f} / "
        f"AIRBORNE {thresholds[Category.AIRBORNE]:.2f}"
    )


def format_category_probs(category_probs: dict[Category, float]) -> str:
    return " ".join(
        f"{category.value}={category_probs[category]:.2f}" for category in Category
    )


def print_window_results(
    path: Path, window_results: list[ClassificationResult], thresholds: Thresholds
) -> None:
    """창마다 카테고리 확률, 판정, top-5를 출력한다 (side effect: 콘솔 출력)."""
    print(f"\n== {path}")
    for index, result in enumerate(window_results):
        judged = judge_category(result.category_probs, thresholds)
        judged_text = judged.value if judged else SKIP_COLUMN
        start_sec = index * config.CLASSIFY_HOP_SEC
        top_text = ", ".join(f"{label} {prob:.2f}" for label, prob in result.top_labels)
        print(
            f"  [{start_sec:4.1f}s] {format_category_probs(result.category_probs)}"
            f" -> {judged_text:8s} | {top_text}"
        )


def clip_category_probs(
    window_results: list[ClassificationResult],
) -> dict[Category, float]:
    """클립 전체의 카테고리 확률 = 창들 중 최댓값. 짧은 이벤트가 무음 창에 묻히지 않게 한다."""
    return {
        category: max(result.category_probs[category] for result in window_results)
        for category in Category
    }


def judged_distribution(
    category_probs_list: list[dict[Category, float]], thresholds: Thresholds
) -> dict[str, float]:
    """판정 카테고리별 비율(%). 키는 impact/airborne/skip."""
    counts = {Category.IMPACT.value: 0, Category.AIRBORNE.value: 0, SKIP_COLUMN: 0}
    for category_probs in category_probs_list:
        judged = judge_category(category_probs, thresholds)
        counts[judged.value if judged else SKIP_COLUMN] += 1
    total = len(category_probs_list)
    return {key: 100.0 * count / total for key, count in counts.items()}


def print_class_summary(
    file_results: list[FileResult], threshold_sets: list[Thresholds]
) -> None:
    """클래스별 확률 중앙값과 임계값별 판정 분포 표를 출력한다 (side effect: 콘솔 출력)."""
    clips_by_class: dict[str, list[dict[Category, float]]] = defaultdict(list)
    windows_by_class: dict[str, list[dict[Category, float]]] = defaultdict(list)
    for file_result in file_results:
        clips_by_class[file_result.class_name].append(
            clip_category_probs(file_result.window_results)
        )
        windows_by_class[file_result.class_name].extend(
            result.category_probs for result in file_result.window_results
        )
    class_names = sorted(clips_by_class)
    name_width = max(len(name) for name in class_names)

    print("\n===== 클립 단위 카테고리 확률 중앙값 (클립 = 창 최댓값) =====")
    header = " ".join(f"{category.value:>9s}" for category in SUMMARY_CATEGORIES)
    print(f"{'class':{name_width}s} {'clips':>5s} {header}")
    for name in class_names:
        medians = " ".join(
            f"{numpy.median([clip[category] for clip in clips_by_class[name]]):9.2f}"
            for category in SUMMARY_CATEGORIES
        )
        print(f"{name:{name_width}s} {len(clips_by_class[name]):5d} {medians}")

    for thresholds in threshold_sets:
        print(f"\n===== 판정 분포 % ({format_thresholds(thresholds)}) =====")
        print(
            f"{'class':{name_width}s} | {'clip impact':>11s} {'airborne':>8s}"
            f" {'skip':>5s} | {'window impact':>13s} {'airborne':>8s} {'skip':>5s}"
        )
        for name in class_names:
            clip = judged_distribution(clips_by_class[name], thresholds)
            window = judged_distribution(windows_by_class[name], thresholds)
            print(
                f"{name:{name_width}s} | {clip['impact']:11.0f} {clip['airborne']:8.0f}"
                f" {clip['skip']:5.0f} | {window['impact']:13.0f}"
                f" {window['airborne']:8.0f} {window['skip']:5.0f}"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wav 파일/폴더 CED 분류 검증")
    parser.add_argument("input_path", type=Path, help="wav 파일 또는 폴더")
    parser.add_argument("--meta", type=Path, help="ESC-50 형식 메타데이터 CSV")
    parser.add_argument("--classes", help="쉼표로 구분한 대상 클래스 (폴더 모드)")
    parser.add_argument(
        "--impact-threshold",
        type=float,
        default=config.CLASS_PROB_THRESHOLD[Category.IMPACT],
        help="창별 출력과 요약 조합에 쓰는 IMPACT 임계값",
    )
    parser.add_argument(
        "--airborne-threshold",
        type=float,
        default=config.CLASS_PROB_THRESHOLD[Category.AIRBORNE],
        help="창별 출력과 요약 조합에 쓰는 AIRBORNE 임계값",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in config.FILE_EVAL_THRESHOLDS),
        help="두 카테고리에 같은 값을 적용해 비교할 후보들 (쉼표 구분)",
    )
    parser.add_argument(
        "--scope", choices=["extended", "legal"], default=config.AIRBORNE_SCOPE
    )
    parser.add_argument(
        "--verbose", action="store_true", help="폴더 모드에서도 창별 결과 출력"
    )
    return parser.parse_args()


def main() -> None:
    """인자에 따라 파일 또는 폴더를 분류하고 결과를 출력한다 (side effect: 파일 읽기, 출력)."""
    arguments = parse_arguments()
    thresholds: Thresholds = {
        Category.IMPACT: arguments.impact_threshold,
        Category.AIRBORNE: arguments.airborne_threshold,
    }
    threshold_sets = [
        uniform_thresholds(float(value)) for value in arguments.thresholds.split(",")
    ] + [thresholds]
    classifier = CedClassifier(airborne_scope=arguments.scope)
    print(f"AIRBORNE scope={arguments.scope}")

    if arguments.input_path.is_file():
        window_results = classify_wav(classifier, arguments.input_path)
        print_window_results(arguments.input_path, window_results, thresholds)
        return

    class_names = load_class_names(arguments.meta) if arguments.meta else None
    wanted_classes = set(arguments.classes.split(",")) if arguments.classes else None
    targets = collect_wav_files(arguments.input_path, class_names, wanted_classes)
    if not targets:
        raise SystemExit("대상 wav 파일이 없습니다 (경로/--meta/--classes 확인)")
    print(f"{len(targets)}개 파일 분류 중...")

    file_results = []
    for path, class_name in targets:
        window_results = classify_wav(classifier, path)
        file_results.append(FileResult(path, class_name, window_results))
        if arguments.verbose:
            print_window_results(path, window_results, thresholds)
    print_class_summary(file_results, threshold_sets)


if __name__ == "__main__":
    main()
