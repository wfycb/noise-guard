"""Pi 클라이언트 전용 상수. 서버의 config.py와 분리한다 (Pi에서 서버 모듈을 import하지 않기 위해)."""

# --- 서버 연결 ---
# README의 직결 랜 예시(노트북 192.168.10.1/24). 실행 시 --server로 바꿀 수 있다.
SERVER_HOST: str = "192.168.10.1"
SERVER_PORT: int = 5000
CLIENT_ID: str = "pi-1"
CONNECT_TIMEOUT_SEC: float = 5.0
HELLO_ACK_TIMEOUT_SEC: float = 5.0
# 이 시간 동안 서버로부터 아무것도 못 받으면 끊긴 것으로 본다. 서버는 PING_INTERVAL_SEC마다 PING을 보낸다.
PEER_TIMEOUT_SEC: float = 6.0
# PEER_TIMEOUT_SEC의 1/3 이하여야 한다(protocol.validate_ping_interval이 확인).
PING_INTERVAL_SEC: float = 2.0
# 연결이 끊기거나 실패하면 이 간격부터 두 배씩 늘려 다시 시도한다(최대 RECONNECT_MAX_SEC).
RECONNECT_INITIAL_SEC: float = 1.0
RECONNECT_MAX_SEC: float = 30.0
# 파일 재생이 끝난 뒤 서버가 남은 오디오를 처리하고 마지막 알림을 보낼 때까지 기다리는 상한.
DRAIN_TIMEOUT_SEC: float = 300.0

# --- 수집 ---
MIC_SAMPLE_RATE: int = 48000
# 48kHz에서 100ms. 메시지 하나 = 마이크 1개의 청크 하나.
CHUNK_SAMPLES: int = 4800
# 방 이름 → sounddevice 장치 인덱스 또는 이름. tools/list_devices.py로 확인해 채운다.
MIC_DEVICES: dict[str, int | str] = {}

# --- 송신 ---
# 네트워크가 막혀도 수집 콜백이 멈추지 않도록 큐 크기를 제한한다. 넘치면 가장 오래된 청크를 버린다.
# 마이크 5개 × 100ms 청크 기준 약 4초 분량.
SEND_QUEUE_MAX_CHUNKS: int = 200
QUEUE_POLL_SEC: float = 0.2

# --- 출력 (LED / 디스플레이) ---
LED_BLINK_COUNT: int = 3  # caution: LED 점멸 횟수
LED_BLINK_INTERVAL_SEC: float = 0.25
DISPLAY_HOLD_SEC: float = 10.0  # warning: 디스플레이 표시 유지 시간
# 서버 연결이 끊긴 동안 "N초째" 표시를 이 간격으로 갱신한다.
DISCONNECT_STATUS_REFRESH_SEC: float = 5.0

# --- 종단 지연 측정 ---
# ALERT의 source_seq로 캡처 시각을 찾기 위해 seq별 캡처 시각을 이 시간만큼 기억한다.
LATENCY_HISTORY_SEC: float = 30.0

# --- 스레드 ---
THREAD_JOIN_TIMEOUT_SEC: float = 5.0
THREAD_NAME_PREFIX: str = "noise-guard-pi"
