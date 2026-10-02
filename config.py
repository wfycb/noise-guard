"""모든 상수. 다른 모듈에는 매직 넘버를 두지 않는다."""

from typing import Literal

from models import Category

# --- 오디오 수집 ---
# 노트북 내장 마이크(WASAPI)는 공유 모드 믹스 포맷인 48kHz만 허용한다 (Step 1에서 확인).
CAPTURE_SAMPLE_RATE: int = 48000
CAPTURE_CHANNELS: int = 1
CAPTURE_DTYPE: str = "float32"
# 분류 창보다 넉넉하게 잡아서 워커가 조금 늦어도 창이 덮어써지지 않게 한다.
RING_BUFFER_SEC: float = 4.0

# --- 분류 (CED) ---
# 모델 크기 → (HF 모델 이름, 커밋). trust_remote_code라서 원격 코드가 바뀌면 동작이 달라질 수 있으므로
# 검증한 커밋에 고정한다. 두 모델 모두 라벨 527개 순서가 같고 pooling=mean(출력이 sigmoid 확률)임을 확인했다.
CED_MODELS: dict[str, tuple[str, str]] = {
    "base": ("mispeech/ced-base", "db3e14a8db4c21b56b165261c39649741a900e7f"),
    "mini": ("mispeech/ced-mini", "26c3ebcae85d4330f4fc26763f029539a3afcda0"),
}
CED_MODEL_SIZE: str = "base"
CED_MODEL_NAME, CED_MODEL_REVISION = CED_MODELS[CED_MODEL_SIZE]
CLASSIFIER_SAMPLE_RATE: int = 16000
CLASSIFY_WINDOW_SEC: float = 2.0
CLASSIFY_HOP_SEC: float = 1.0
TOP_K_LABELS: int = 5
# "legal": TV·음향기기(음악)만 공기전달로 본다 (법 기준).
# "extended": 말소리·개·청소기 등도 포함하는 배려 확장 모드.
AIRBORNE_SCOPE: Literal["extended", "legal"] = "extended"

# --- 판단 ---
# 카테고리별 판정 임계값: 해당 카테고리 확률이 이 값 이상이어야 그 카테고리로 판정한다.
CLASS_PROB_THRESHOLD: dict[Category, float] = {
    # ESC-50 근거리 녹음 기준 임시값, USB 마이크와 미니어처로 재조정 필요.
    # 0.3 대비 발소리 검출률이 높고(클립 82% vs 72%) 비충격음 오탐은 낮게 유지됨.
    Category.IMPACT: 0.2,
    # ESC-50 근거리 녹음 기준 임시값, USB 마이크와 미니어처로 재조정 필요.
    # 무음에서도 Music이 약 0.11~0.13 나오므로 그보다 충분히 높게 잡는다.
    Category.AIRBORNE: 0.3,
}

# --- 층간소음 기준 (2023 개정 공동주택 층간소음 기준), dB(A) ---
IMPACT_LEQ_LIMIT_DAY_DB: float = 39.0  # 직접충격 1분 Leq, 주간
IMPACT_LEQ_LIMIT_NIGHT_DB: float = 34.0  # 직접충격 1분 Leq, 야간
IMPACT_LMAX_LIMIT_DAY_DB: float = 57.0  # 직접충격 Lmax, 주간
IMPACT_LMAX_LIMIT_NIGHT_DB: float = 52.0  # 직접충격 Lmax, 야간
AIRBORNE_LEQ_LIMIT_DAY_DB: float = 45.0  # 공기전달 5분 Leq, 주간
AIRBORNE_LEQ_LIMIT_NIGHT_DB: float = 40.0  # 공기전달 5분 Leq, 야간

# 주간 06–22시, 야간 22–06시 (로컬 시각 기준).
DAY_START_HOUR: int = 6
NIGHT_START_HOUR: int = 22
# Asia/Seoul. 서머타임이 없어 고정 오프셋으로 충분하고,
# zoneinfo는 Windows에서 tzdata 패키지가 추가로 필요해 쓰지 않는다.
LOCAL_UTC_OFFSET_HOURS: int = 9

# --- 판단 규칙 (R1~R4) ---
LMAX_COUNT_TO_WARN: int = 3  # Lmax 기준 초과가 시간창 내 이 횟수 이상이면 R2 경고
LMAX_WINDOW_SEC: float = 3600.0  # 법 기준: 1시간에 3회 이상
IMPACT_LEQ_WINDOW_SEC: float = 60.0  # 법 기준: 1분 Leq
AIRBORNE_LEQ_WINDOW_SEC: float = 300.0  # 법 기준: 5분 Leq
MERGE_GAP_SEC: float = 2.0  # 같은 카테고리 프레임 간격이 이 이하면 같은 이벤트
# 표시·문서용 계산값 (판단 로직은 MERGE_GAP_SEC를 쓴다).
# 분류 창이 hop보다 길어서 소리가 끝난 뒤에도 IMPACT 판정이 이어지므로,
# 실제 무음이 이 길이 이하이면 앞뒤 충격이 같은 이벤트로 합쳐진다.
EFFECTIVE_MERGE_GAP_SEC: float = MERGE_GAP_SEC + (
    CLASSIFY_WINDOW_SEC - CLASSIFY_HOP_SEC
)
COOLDOWN_SEC: float = 30.0  # 같은 규칙 warning 재발령 금지 시간

# 시연 모드: 짧은 데모 안에서 규칙이 동작하는 것을 보여주기 위해 시간창만 줄인다.
DEMO_MODE: bool = False
DEMO_LMAX_WINDOW_SEC: float = 60.0
DEMO_IMPACT_LEQ_WINDOW_SEC: float = 20.0
DEMO_AIRBORNE_LEQ_WINDOW_SEC: float = 60.0
DEMO_COOLDOWN_SEC: float = 10.0

# 프레임 하나가 대표하는 시간. Leq를 시간 가중 에너지로 계산할 때 쓴다.
FRAME_SEC: float = CLASSIFY_HOP_SEC
# 카테고리별 Leq는 창 전체 길이로 나누고(사양 9.5 (a)), 해당 카테고리가 아닌 구간은
# 이 레벨로 채운다. "분류된 소음의 기여분만 본다"는 설계라 다른 구간의 기여는 0에 가까워야 하므로
# 기준값(34~57 dB(A))보다 충분히 낮은 0 dB(A)로 둔다 (60초 창 전체를 채워도 기여 0 dB(A)).
LEQ_FLOOR_DB: float = 0.0

# --- 조용할 때 분류 건너뛰기 ---
# 마이크의 분류 창(2초) Leq(추정 dB(A))가 이 값보다 낮으면 그 마이크는 CED에 넣지 않는다(확률 0).
# 1초가 아니라 분류 창 전체로 보는 이유: 창 안에 큰 소리가 있으면 CED가 IMPACT로 볼 수 있으므로
# 그런 프레임을 건너뛰면 이벤트 병합이 달라져 --no-skip과 알림이 달라진다.
# 판단 기준 중 가장 낮은 값(야간 R3 34)에서 SKIP_MARGIN_DB를 뺀 값 이하여야 한다. 기준 근처 소리를
# 건너뛰면 R3/R4 Leq가 과소평가되기 때문이다(classifier.validate_skip_gate가 시작할 때 확인).
# !!! 보정 전 임시 오프셋(+100 dB) 기준이라 이 dB 값 자체에는 의미가 없다. 소음계 보정 후 다시 확인 !!!
SKIP_CLASSIFY_BELOW_DB: float = 24.0
SKIP_MARGIN_DB: float = 10.0
QUIET_TOP_LABEL: str = "(skipped: quiet)"

# --- tools/classify_file.py ---
# 두 카테고리에 같은 값을 적용해 비교해 볼 후보값들 (config 조합은 항상 함께 출력).
FILE_EVAL_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

# --- 레벨 ---
# 무음에서 log10(0) = -inf 를 막기 위한 하한. 약 -200 dBFS에 해당한다.
DBFS_EPSILON: float = 1e-10
# IEC 61672 A특성 아날로그 전달함수 상수 (Hz, dB).
A_WEIGHTING_F1: float = 20.598997
A_WEIGHTING_F2: float = 107.65265
A_WEIGHTING_F3: float = 737.86223
A_WEIGHTING_F4: float = 12194.217
A_WEIGHTING_A1000_DB: float = 1.9997
# 소음계 Fast 시간가중(125ms)을 블록 RMS로 근사한다.
FAST_BLOCK_SEC: float = 0.125

# --- 보정 (dBFS(A) → 추정 dB(A)) ---
# !!! 임시값: 소음계 보정 전. 실제 SPL이 아니다 !!!
# Step 1에서 내장 마이크로 가까이서 말했을 때 약 -40 dBFS가 나왔고,
# 일반 대화를 약 60 dB(A)로 보이게 하려고 100 dB로 잡았다. tools/calibrate.py로 대체한다.
DEFAULT_CALIBRATION_OFFSET_DB: float = 100.0
# 방 이름 → 오프셋. 서버 시작 시 calibration.json이 있으면 calibration.apply_calibration이 채운다.
# 여기 없는 방은 DEFAULT(임시)를 쓰고 "보정 안 됨"으로 표시한다.
CALIBRATION_OFFSET_DB: dict[str, float] = {}
CALIBRATION_FILE: str = "calibration.json"
# 배경 소음·핑크노이즈를 측정할 시간. 소음계 Leq(또는 Fast 값 여러 번 평균)를 읽는 시간과 맞춘다.
CALIBRATION_MEASURE_SEC: float = 10.0
# 볼륨을 바꿔 여러 번 잰 오프셋의 차이가 이보다 크면 경고한다(마이크 AGC가 켜져 있으면 커짐).
CALIBRATION_MAX_SPREAD_DB: float = 2.0
# 배경 소음이 최저 판단 기준에서 이 값 이내면 오탐 위험으로 경고한다.
CALIBRATION_BACKGROUND_NEAR_LIMIT_DB: float = 5.0
# 권장 게이트 = min(배경 + 이 값, 최저 기준 − SKIP_MARGIN_DB). 배경 소음 프레임은 건너뛰되 여유를 둔다.
CALIBRATION_GATE_HEADROOM_DB: float = 3.0

# --- 마이크 매핑 (main.py --source mic) ---
# 방 이름 → 장치 인덱스 또는 이름. 같은 모델 USB 마이크는 이름이 같을 수 있으니
# tools/list_devices.py 로 인덱스를 확인해 넣는다. 지금은 내장 마이크 하나뿐이다.
MIC_DEVICES: dict[str, int | str] = {"거실": 18}

# --- 네트워크 서버 (main.py --source network) ---
# 모든 인터페이스에서 받는다. 배포 시에는 Pi와 직결한 랜 인터페이스 IP로 바꾸는 것을 권장한다.
SERVER_BIND_HOST: str = "0.0.0.0"
SERVER_PORT: int = 5000
# 접속 직후 HELLO를 이 시간 안에 보내지 않으면 끊는다.
HELLO_TIMEOUT_SEC: float = 5.0
# 이 시간 동안 아무 메시지도 받지 못하면 끊긴 것으로 본다. 오디오가 계속 오므로 정상 연결에서는 걸리지 않는다.
PEER_TIMEOUT_SEC: float = 6.0
# 서버도 이 간격으로 PING을 보낸다. 알림이 드물어도 Pi가 끊김을 감지할 수 있게 하기 위해서다.
# PEER_TIMEOUT_SEC의 1/3 이하여야 한다(protocol.validate_ping_interval이 확인).
PING_INTERVAL_SEC: float = 2.0
# 다른 마이크는 tick 분량이 도착했는데 한 마이크만 이 시간 넘게 늦으면, 그 tick에서 그 마이크를 뺀다.
MIC_STALL_TIMEOUT_SEC: float = 2.0
# 도착했지만 아직 처리하지 못한 오디오의 마이크별 최대 길이. 넘으면 서버가 너무 밀린 것이라 연결을 끊는다.
NETWORK_BACKLOG_MAX_SEC: float = 60.0
# 서버 스레드가 종료 요청을 확인하는 주기. 짧을수록 종료가 빠르고 CPU를 조금 더 쓴다.
SERVER_THREAD_POLL_SEC: float = 0.2
# 종료 시 스레드 join 대기 상한. 넘으면 남은 스레드를 경고로 남긴다.
THREAD_JOIN_TIMEOUT_SEC: float = 5.0
THREAD_NAME_PREFIX: str = "noise-guard"
# 마이크가 이만큼 연속으로 tick에서 빠지면 "마이크 이상"으로 경고하고 Pi에 STATUS를 보낸다.
MIC_MISSING_WARN_TICKS: int = 3

# --- tools/classify_live.py ---
# 노트북 내장 마이크 배열의 WASAPI 장치 번호. 장치를 꽂고 빼면 바뀔 수 있으니
# tools/list_devices.py 로 다시 확인한다.
LIVE_DEFAULT_DEVICE: int = 18
