"""ESC-50 클립을 이어 붙여 파일 소스용 시나리오 wav를 만든다 (side effect: 파일 읽기/쓰기).

실행 (프로젝트 루트에서):
    .venv\\Scripts\\python -m tools.make_scenario [--gain-db 0] [--only S1,S5]

클립은 tools/classify_file.py 결과에서 해당 카테고리 확률이 높았던 것을 골랐다.
분류 정확도가 아니라 파이프라인(병합·카운트·판단·위치)을 검증하려는 것이기 때문이다.
레벨은 클립 원래 진폭을 유지한다 (float32 wav라 1.0을 넘어도 잘리지 않는다).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy
from scipy.io import wavfile

from capture import read_wav_mono

DEFAULT_ESC50_AUDIO_DIR = Path("data/esc50/audio")
DEFAULT_OUTPUT_DIR = Path("data/scenarios")
# 분류 창(2초)이 찰 때까지 첫 프레임이 나오지 않으므로 앞에 무음을 둔다.
LEAD_SILENCE_SEC = 2.0
TAIL_SILENCE_SEC = 3.0
EVENT_GAP_SEC = 5.0  # 사양 S1: 발소리 클립 사이 5초 무음
FAR_ROOM_GAIN_DB = -20.0  # S5: 소리가 난 방에서 먼 마이크를 흉내 낸 감쇠
# S3b: 원래 진폭 그대로 섞으면 물소리(RMS 약 -13 dBFS)가 발소리(약 -34 dBFS)를 가려서
# CED가 발소리를 못 듣는다. 두 소리가 비슷한 크기일 때 판정 로직을 보기 위해 물소리만 낮춘다.
WATER_UNDER_FOOTSTEPS_GAIN_DB = -20.0

FOOTSTEPS = ("1-155858-A-25.wav", "1-155858-D-25.wav", "1-155858-E-25.wav")
TOILET_FLUSH = ("3-112356-A-18.wav", "4-141365-A-18.wav")
POURING_WATER = ("3-161500-A-17.wav", "2-102414-E-17.wav")
DOG = (
    "3-155312-A-0.wav",
    "3-180977-A-0.wav",
    "5-208030-A-0.wav",
    "5-203128-A-0.wav",
    "1-32318-A-0.wav",
    "1-97392-A-0.wav",
    "3-155312-A-0.wav",
)
DOOR_KNOCK = (
    "2-140841-A-30.wav",
    "4-182041-A-30.wav",
    "2-118625-A-30.wav",
    "3-144510-A-30.wav",
    "4-211502-A-30.wav",
)
CRYING_BABY = (
    "3-152007-E-20.wav",
    "3-152007-A-20.wav",
    "2-107351-B-20.wav",
    "2-151079-A-20.wav",
    "1-60997-A-20.wav",
    "2-80482-A-20.wav",
    "3-152007-E-20.wav",
)


@dataclass(frozen=True)
class Silence:
    duration_sec: float


@dataclass(frozen=True)
class Clip:
    filename: str
    gain_db: float = 0.0


@dataclass(frozen=True)
class Mix:
    """여러 클립을 같은 시점에 겹친다 (길이는 가장 긴 클립 기준)."""

    clips: tuple[Clip, ...]


Segment = Silence | Clip | Mix
# 시나리오 이름 → {출력 파일 접미사(방 이름): 구간 목록}
SCENARIOS: dict[str, dict[str, list[Segment]]] = {
    "S1_footsteps_x3": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            Clip(FOOTSTEPS[0]),
            Silence(EVENT_GAP_SEC),
            Clip(FOOTSTEPS[1]),
            Silence(EVENT_GAP_SEC),
            Clip(FOOTSTEPS[2]),
            Silence(TAIL_SILENCE_SEC),
        ]
    },
    "S2_water_only": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            Clip(TOILET_FLUSH[0]),
            Clip(POURING_WATER[0]),
            Clip(TOILET_FLUSH[1]),
            Clip(POURING_WATER[1]),
            Silence(TAIL_SILENCE_SEC),
        ]
    },
    "S3_water_plus_footsteps": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            Mix((Clip(TOILET_FLUSH[0]), Clip(FOOTSTEPS[0]))),
            Silence(EVENT_GAP_SEC),
            Mix((Clip(POURING_WATER[0]), Clip(FOOTSTEPS[2]))),
            Silence(TAIL_SILENCE_SEC),
        ]
    },
    "S3b_quiet_water_plus_footsteps": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            Mix(
                (
                    Clip(TOILET_FLUSH[0], WATER_UNDER_FOOTSTEPS_GAIN_DB),
                    Clip(FOOTSTEPS[0]),
                )
            ),
            Silence(EVENT_GAP_SEC),
            Mix(
                (
                    Clip(POURING_WATER[0], WATER_UNDER_FOOTSTEPS_GAIN_DB),
                    Clip(FOOTSTEPS[2]),
                )
            ),
            Silence(TAIL_SILENCE_SEC),
        ]
    },
    "S4_dog_and_baby": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            *(
                Clip(filename)
                for pair in zip(DOG, CRYING_BABY, strict=True)
                for filename in pair
            ),
            Silence(TAIL_SILENCE_SEC),
        ]
    },
    "S5_two_rooms": {
        "거실": [
            Silence(LEAD_SILENCE_SEC),
            Clip(FOOTSTEPS[0]),
            Silence(EVENT_GAP_SEC),
            Clip(FOOTSTEPS[2], FAR_ROOM_GAIN_DB),
            Silence(TAIL_SILENCE_SEC),
        ],
        "안방": [
            Silence(LEAD_SILENCE_SEC),
            Clip(FOOTSTEPS[0], FAR_ROOM_GAIN_DB),
            Silence(EVENT_GAP_SEC),
            Clip(FOOTSTEPS[2]),
            Silence(TAIL_SILENCE_SEC),
        ],
    },
}


def spaced_clips(filenames: tuple[str, ...], gap_sec: float) -> list[Segment]:
    """앞 무음 + (클립, 간격 무음) 반복 + 뒤 무음."""
    segments: list[Segment] = [Silence(LEAD_SILENCE_SEC)]
    for filename in filenames:
        segments += [Clip(filename), Silence(gap_sec)]
    return segments + [Silence(TAIL_SILENCE_SEC)]


# Step 6 부하 확인용 (회귀 테스트 대상 아님): 방 5개가 동시에 서로 다른 소리를 낸다.
# 방마다 클립 사이 간격을 달리해 이벤트 시점이 겹치기도 하고 엇갈리기도 하게 했다.
LOAD_SCENARIOS: dict[str, dict[str, list[Segment]]] = {
    "S6_five_rooms": {
        "거실": spaced_clips(FOOTSTEPS * 3, gap_sec=2.0),
        "안방": spaced_clips(DOG, gap_sec=3.0),
        "주방": spaced_clips(TOILET_FLUSH + POURING_WATER, gap_sec=8.0),
        "서재": spaced_clips(CRYING_BABY, gap_sec=4.0),
        "현관": spaced_clips(DOOR_KNOCK, gap_sec=6.0),
    }
}


def db_to_amplitude(gain_db: float) -> float:
    return 10.0 ** (gain_db / 20.0)


class ClipLoader:
    """ESC-50 클립을 읽고 샘플레이트가 모두 같은지 확인한다 (side effect: 파일 읽기)."""

    def __init__(self, audio_dir: Path) -> None:
        self._audio_dir = audio_dir
        self.sample_rate: int | None = None

    def load(self, clip: Clip, extra_gain_db: float) -> numpy.ndarray:
        samples, sample_rate = read_wav_mono(self._audio_dir / clip.filename)
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif sample_rate != self.sample_rate:
            raise ValueError(
                f"{clip.filename}: 샘플레이트 {sample_rate} ≠ {self.sample_rate}"
            )
        return samples * db_to_amplitude(clip.gain_db + extra_gain_db)


def render_segments(
    segments: list[Segment], loader: ClipLoader, extra_gain_db: float
) -> numpy.ndarray:
    """구간들을 순서대로 이어 붙인 샘플 배열을 만든다."""
    pieces: list[numpy.ndarray] = []
    for segment in segments:
        if isinstance(segment, Clip):
            pieces.append(loader.load(segment, extra_gain_db))
        elif isinstance(segment, Mix):
            parts = [loader.load(clip, extra_gain_db) for clip in segment.clips]
            mixed = numpy.zeros(max(len(part) for part in parts), dtype=numpy.float32)
            for part in parts:
                mixed[: len(part)] += part
            pieces.append(mixed)
        else:
            if loader.sample_rate is None:
                raise ValueError(
                    "무음 길이를 정하려면 클립이 먼저 하나 읽혀 있어야 합니다"
                )
            pieces.append(
                numpy.zeros(
                    round(segment.duration_sec * loader.sample_rate),
                    dtype=numpy.float32,
                )
            )
    return numpy.concatenate(pieces).astype(numpy.float32)


def first_clip_filename(segments: list[Segment]) -> str:
    for segment in segments:
        if isinstance(segment, Clip):
            return segment.filename
        if isinstance(segment, Mix):
            return segment.clips[0].filename
    raise ValueError("클립이 없는 시나리오입니다")


def write_scenario(
    name: str,
    tracks: dict[str, list[Segment]],
    audio_dir: Path,
    output_dir: Path,
    extra_gain_db: float,
) -> list[Path]:
    """시나리오 하나를 방별 wav로 저장하고 경로를 반환한다 (side effect: 파일 읽기/쓰기)."""
    written = []
    for room_name, segments in tracks.items():
        loader = ClipLoader(audio_dir)
        # 무음 길이를 샘플 수로 바꾸려면 샘플레이트가 필요하므로 첫 클립을 미리 읽는다.
        loader.load(Clip(first_clip_filename(segments)), extra_gain_db)
        samples = render_segments(segments, loader, extra_gain_db)
        path = output_dir / f"{name}_{room_name}.wav"
        wavfile.write(path, loader.sample_rate, samples)
        peak = float(numpy.max(numpy.abs(samples)))
        print(
            f"{path}  {len(samples) / loader.sample_rate:5.1f}s "
            f"@ {loader.sample_rate} Hz, peak {peak:.2f}"
            + ("  (1.0 초과: float wav라 잘리지 않음)" if peak > 1.0 else "")
        )
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="ESC-50 기반 시나리오 wav 생성")
    parser.add_argument("--esc50-dir", type=Path, default=DEFAULT_ESC50_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gain-db", type=float, default=0.0, help="모든 클립에 더할 게인(dB)"
    )
    parser.add_argument("--only", help="만들 시나리오 접두어 (예: S1,S5)")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    wanted_prefixes = arguments.only.split(",") if arguments.only else None
    for name, tracks in {**SCENARIOS, **LOAD_SCENARIOS}.items():
        if wanted_prefixes and not any(name.startswith(p) for p in wanted_prefixes):
            continue
        write_scenario(
            name, tracks, arguments.esc50_dir, arguments.output_dir, arguments.gain_db
        )


if __name__ == "__main__":
    main()
