"""Step 1: 단일 마이크 실시간 CED 분류 확인 (side effect: 오디오 장치 접근, 콘솔 출력).

실행 (프로젝트 루트에서):
    .venv\\Scripts\\python -m tools.classify_live [--device 18] [--duration 30]
"""

import argparse
import time

import numpy
import sounddevice

import config
from capture import MicSource
from classifier import CedClassifier, resample_for_classifier
from level import LevelMeter, leq_db, to_estimated_dba

LIVE_MIC_NAME = "live"
REFERENCE_NOTICE = (
    "※ dB 값은 참고용: dBFS(A)는 A특성 적용값, dB(A)는 보정 전 임시 오프셋"
    f"(+{config.DEFAULT_CALIBRATION_OFFSET_DB:.0f} dB)을 더한 추정치로 실제 SPL이 아님. "
    "노트북 내장 마이크 배열은 OS/드라이버 DSP(노이즈 억제·AGC)가 적용될 수 있음. "
    "이 도구는 분류 동작 확인용."
)


def parse_device(value: str) -> int | str:
    """숫자면 장치 인덱스, 아니면 장치 이름 일부로 해석한다."""
    return int(value) if value.isdigit() else value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="단일 마이크 CED 실시간 분류 확인")
    parser.add_argument(
        "--device",
        type=parse_device,
        default=config.LIVE_DEFAULT_DEVICE,
        help="입력 장치 인덱스 또는 이름 (tools/list_devices.py 참고)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="실행 시간(초). 생략하면 Ctrl+C까지 실행",
    )
    return parser.parse_args()


def print_output_format(probabilities: numpy.ndarray) -> None:
    """첫 추론 결과의 shape과 값 범위를 출력한다 (side effect: 콘솔 출력)."""
    within_unit_range = bool(probabilities.min() >= 0.0 and probabilities.max() <= 1.0)
    print(
        f"[출력 형식] shape={probabilities.shape} "
        f"min={probabilities.min():.3e} max={probabilities.max():.4f} "
        f"0~1 범위={within_unit_range}"
    )
    # CED 출력은 이미 sigmoid가 적용된 확률이므로 다시 적용하지 않는다.
    # 근거: modeling_ced.py의 CedForAudioClassification.forward_head가
    # `self.outputlayer(x).sigmoid()`를 반환한다.
    print("[출력 형식] CED .logits 는 sigmoid 적용된 확률 → 재적용하지 않음")


def print_latency_summary(latencies_ms: list[float], overflow_count: int) -> None:
    """추론 지연 통계를 출력한다. 첫 회는 워밍업이라 따로 표시한다 (side effect: 콘솔 출력)."""
    print("\n===== 요약 =====")
    print(f"추론 횟수: {len(latencies_ms)}, overflow: {overflow_count}")
    if not latencies_ms:
        return
    print(f"첫 추론(워밍업): {latencies_ms[0]:.1f} ms")
    steady_latencies = latencies_ms[1:] or latencies_ms
    print(
        f"이후 평균: {numpy.mean(steady_latencies):.1f} ms, "
        f"최대: {numpy.max(steady_latencies):.1f} ms"
    )


def run_live_classification(device: int | str, duration_sec: float | None) -> None:
    """장치를 열고 hop마다 분류 결과를 출력한다 (side effect: 장치 접근, 콘솔 출력)."""
    print("CED 로딩 중...")
    classifier = CedClassifier()
    print(f"CED 로딩 완료 (torch device={classifier.device})")

    mic_stream = MicSource(LIVE_MIC_NAME, device)
    window_samples = int(config.CLASSIFY_WINDOW_SEC * mic_stream.sample_rate)
    level_meter = LevelMeter(mic_stream.sample_rate)
    total_read = 0
    print(
        f"장치: {sounddevice.query_devices(device)['name']} @ {mic_stream.sample_rate} Hz"
    )
    print(REFERENCE_NOTICE)

    latencies_ms: list[float] = []
    mic_stream.start()
    start_time = time.monotonic()
    next_tick = start_time + config.CLASSIFY_HOP_SEC
    try:
        while duration_sec is None or time.monotonic() - start_time < duration_sec:
            time.sleep(max(0.0, next_tick - time.monotonic()))
            next_tick += config.CLASSIFY_HOP_SEC
            # 레벨은 새 샘플만 한 번씩 필터에 넣어야 A특성 필터 상태가 이어진다.
            new_samples, total_read = mic_stream.read_new(total_read)
            block_levels = level_meter.push(new_samples)
            window = mic_stream.latest(window_samples)
            if window is None or not block_levels:
                continue

            resampled_window = resample_for_classifier(window, mic_stream.sample_rate)
            inference_start = time.perf_counter()
            result = classifier.classify({LIVE_MIC_NAME: resampled_window})[
                LIVE_MIC_NAME
            ]
            latency_ms = (time.perf_counter() - inference_start) * 1000.0
            latencies_ms.append(latency_ms)

            if len(latencies_ms) == 1:
                print_output_format(result.probabilities[numpy.newaxis, :])

            frame_leq_dbfs_a = leq_db(block_levels)
            frame_lmax_dbfs_a = max(block_levels)
            frame_leq_dba = to_estimated_dba(frame_leq_dbfs_a, LIVE_MIC_NAME)
            frame_lmax_dba = to_estimated_dba(frame_lmax_dbfs_a, LIVE_MIC_NAME)
            top_text = ", ".join(
                f"{label} {prob:.2f}" for label, prob in result.top_labels
            )
            print(
                f"{time.strftime('%H:%M:%S')} | {latency_ms:6.1f} ms | "
                f"Leq {frame_leq_dbfs_a:6.1f} / Lmax {frame_lmax_dbfs_a:6.1f} dBFS(A)"
                f" → 추정 {frame_leq_dba:5.1f} / {frame_lmax_dba:5.1f} dB(A)(임시)"
                f" | {top_text}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        mic_stream.close()
        print_latency_summary(latencies_ms, mic_stream.overflow_count)


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        run_live_classification(arguments.device, arguments.duration)
    except sounddevice.PortAudioError as error:
        raise SystemExit(f"입력 장치 {arguments.device!r}를 열 수 없습니다: {error}")
