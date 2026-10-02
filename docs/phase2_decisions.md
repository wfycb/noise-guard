# Phase 2 사양 변경 기록

Phase 2(라즈베리파이 ↔ 노트북) 작업에서 원래 사양과 다르게 정하거나 사양에 없던 것을 정한 내용과 그 이유를
단계별로 적는다. 모두 단계 보고에서 사용자 확인을 받은 결정이다.

## N0 현황 파악과 설계

- **NetworkSource를 파이프라인 변경 없이 끼움:** `FileSource`가 `advance(hop)`로 파일 데이터를 공급하듯,
  `NetworkSource.advance(hop)`가 수신 버퍼에서 정확히 hop 분량을 꺼낸다. `realtime_pacing=False`로 돌리면 프레임
  시각이 오디오 샘플 수 기준이 된다.
- **Pi 쪽 wav 읽기:** Pi는 scipy를 쓸 수 없고 시나리오 wav는 float32다. `make_scenario`를 int16으로 바꾸는 대신
  `pi_client`에 numpy RIFF 파서(PCM16, float32, EXTENSIBLE)를 둔다. float → int16 변환에서 잘린 샘플은 세어 로그로 남긴다.
- **네트워크 경로의 동등성 기준:** int16 양자화 때문에 dB 값이 비트 단위로 같지 않으므로, 직접 실행과의 비교는
  알림의 규칙·횟수·위치로 한다.

## N1 통신 규격

- **규격 상수 위치:** `MAX_MESSAGE_BYTES`, 허용 샘플레이트, `chunk_samples` 범위 등은 `config.py`가 아니라
  `protocol.py`에 둔다. `pi_client`는 `config.py`를 import할 수 없고, 양쪽이 반드시 같은 값을 써야 하기 때문이다.
- **AUDIO 헤더:** `<BIIH`를 유지하고 예약 필드를 `flags`로 쓴다(bit0 = Pi 콜백 overflow). 예약 비트가 켜져
  있으면 규격 위반으로 본다. 길이 접두는 big-endian, AUDIO 헤더와 PCM은 little-endian인 것을 문서에 명시한다.
- **청크 범위:** `chunk_samples`는 160~65535. 하한은 메시지 수 폭증을 막기 위한 값(16kHz 10ms), 상한은 헤더의 uint16.
- **종단 지연 측정:** ALERT에 `source_mic_index`·`source_seq`를 실어 Pi가 자기 시계로 계산한다(시계 동기화 불필요).
  Pi는 최근 `LATENCY_HISTORY_SEC`(30초)의 캡처 시각만 기억하고, 범위 밖은 "측정 불가"로 로그에 남긴다.

## N2 서버 수신

- **오류 정책 분리:** `FramingError`(메시지 경계가 깨짐)는 연결을 끊고, `PayloadError`(경계는 정상, 내용 오류)는
  그 메시지만 무시한다.
- **수신 타임아웃 분리:** 사양의 "PING_INTERVAL_SEC 동안 메시지가 없으면 끊김" 대신 `PEER_TIMEOUT_SEC`(6초)를
  따로 두고, PING 간격은 그 1/3 이하로 한다(N4에서 검증 추가). 간격과 타임아웃이 같으면 정상 연결도 경계에서 오판할 수 있다.
- **마이크 지연 처리:** 일부 마이크만 늦으면 다른 마이크 데이터가 도착한 시각부터 `MIC_STALL_TIMEOUT_SEC` 뒤에
  그 마이크를 빼고 진행하고, 늦게 온 구간은 버리고 센다. 모든 마이크가 늦으면 기다렸다가 밀린 tick을 순서대로 처리한다.
- **정상 종료:** Pi가 송신만 닫으면 입력이 끝난 것으로 보고, 남은 tick을 처리하고 마지막 알림까지 보낸 뒤 연결을 닫는다.
- **torch 지연 import:** `classifier.py`는 `CedClassifier`를 만들 때만 torch·transformers를 import한다. 가짜 분류기
  테스트에서 수 초짜리 import를 피하기 위해서다.

## N3 명령 회신

- **`AlertSink.emit(alerts, frame)`:** 사양의 `emit(alerts)`에 `frame`을 더했다. ALERT JSON의 `label`, `label_ko`,
  `source_seq`는 프레임 정보가 있어야 만들 수 있다. `run_sources`의 `notifier` 인자는 이름을 유지하고 출력 하나 또는
  목록을 받는다(기존 시나리오 테스트를 바꾸지 않기 위해).
- **마이크 빠짐 표시:** `MIC_MISSING_WARN_TICKS`(3) 연속으로 빠지면 서버 콘솔 경고와 STATUS 메시지(타입 7)를 보낸다.
- **지연 측정 시계:** `time.perf_counter()`를 쓴다. Windows의 `time.monotonic()`은 약 15.6ms 단위라 수십 ms 지연
  측정에 거칠다.

## N4 견고성

- **커밋 전 검사:** `tools/check.ps1`(black → ruff → pytest → pytest -m slow, 실패 시 즉시 종료). `pytest | tail`로
  종료 코드가 가려져 실패한 채 커밋된 일이 있어서 만들었다.
- **분류 창이 차기 전 tick은 마이크 빠짐에서 제외**한다.
- **`--stall` 시뮬레이션:** 파일 위치와 프로토콜 seq를 분리해, 멈춘 구간의 청크는 버리고 seq는 계속 늘린다.
- **재접속 세션의 시작 시각 보정:** 재접속한 세션의 시작 시각이 이전 세션 마지막 프레임보다 앞서면 그 뒤로
  옮긴다. 판단 엔진은 시각이 앞으로만 간다고 가정한다(실시간에서는 일어나지 않고 고속 재생에서만 생김).
- **`PiClient` 클래스 기본값은 재접속 끔, CLI 기본값은 켬:** 기존 테스트가 한 번만 연결하는 것을 전제로 한다.

## N5 CED 건너뛰기와 모델

- **게이트 판정 구간: 1초 Leq → 분류 창(2초) Leq.** CED는 2초 창을 본다. 1초 Leq로 판정하면 큰 소리가 막 끝난
  프레임(창 안에는 그 소리가 있어 CED가 IMPACT로 보는 프레임)까지 건너뛰어 이벤트 병합이 달라지고, 알림이
  `--no-skip`과 달라졌다(마이크 1개 S6: base R1 +1, mini R1 +1·R2 +1). 판정 구간을 분류 창과 맞추면 프레임별
  판정과 알림이 같아진다.
- **`FrameProducer`의 게이트 기본값은 None(건너뛰기 없음):** 기존 시나리오 테스트가 그대로 돌게 하고, 실제
  진입점(`main.py`)이 config 값을 넘긴다.
- **세션 첫 프레임의 초기 STATUS:** STATUS는 상태가 바뀔 때만 보내지만, 재접속한 Pi가 마이크 상태를
  "확인 불가"로 둔 채 머물지 않도록 세션 첫 프레임에서 한 번 보낸다.
- **Pi 표시 상태 합치기:** 서버 연결 상태와 마이크 상태를 `DisplayStatus` 하나로 합쳐 `show_status()`에 넘긴다.
- **기본 모델은 base.** mini는 약 2배 빠르지만 문 삐걱 → AIRBORNE 오분류가 늘었다(45% → 62%).
  `--model mini`로 고를 수 있다. 두 모델 모두 라벨 순서가 같고 `pooling=mean`(출력이 sigmoid 확률)임을 확인했다.

## N6 보정

- **배경 소음 측정 단계 추가:** 핑크노이즈 전에 배경 Leq를 재서 `background_leq_db`로 저장하고, 게이트·기준과
  비교해 경고한다. 게이트는 자동으로 바꾸지 않고 권장값만 출력한다.
- **보정 신호 크기 검사:** 단계마다 핑크노이즈 Leq − 배경 Leq가 `CALIBRATION_MIN_SNR_DB`(10) 미만이면 경고하고
  재측정 여부를 묻는다.
- **calibration.json 형식:** 사양의 방 이름 → 결과 형식을 따른다(이전 뼈대의 `version`/`mics` 감싸기는 쓰지 않음).
- **calibration.json은 git에 넣지 않음:** 장비·위치별 값이다. 형식은 `calibration.example.json`으로 커밋한다.
  테스트가 프로젝트 루트에 이 파일을 만들면 실패하게 했다.
- **처리 지연 감시:** 처리 지연이 hop을 넘은 tick은 경고 로그를 남기고 요약에 횟수·최대값을 넣는다.

## N7 이벤트 녹음

- **분류 창 이전 tick의 오디오도 녹음 버퍼로:** 마이크 빠짐 집계에서는 빼지만, 앞 2초 버퍼에는 넣는다.
- **앞부분이 모자라면 무음 채움:** 시작 직후 알림은 앞 2초가 모자라므로 무음으로 채워 구간 시각을 맞추고
  `pre_padded_sec`에 기록한다.
- **폴더 이름 충돌:** 같은 이름이 있으면 `_2`, `_3`을 붙이고 기존 폴더는 덮어쓰지 않는다. 존재 확인과 생성 사이에
  다른 프로세스가 같은 이름을 만들 수 있으므로, 폴더 생성 자체를 시도하고 이미 있으면 다음 번호로 넘어간다.

## N8 정리

- **README를 Phase 2 기준으로 재구성:** 구조도, 팀 공통 dB 정의, 통신 규격, 실행 방법, 직결 랜, Pi 백그라운드 실행,
  보정, 이벤트 녹음(개인정보 안내 포함), 시연 체크리스트, 알려진 한계, 남은 작업, 팀원 기여.
- **확인하지 못한 명령은 "확인 필요"로 표기:** Windows 인터페이스 이름, Pi의 NetworkManager 연결 이름, systemd 서비스
  설치는 실기기에서 확인해야 한다.
