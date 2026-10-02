"""라즈베리파이 클라이언트: 마이크 수집 → PCM 전송, 서버 명령 수신 → LED/디스플레이 출력.

numpy, sounddevice, 표준 라이브러리, 그리고 저장소 루트의 protocol.py만 import한다.
torch·transformers·scipy와 서버 모듈(config, capture, classifier 등)은 import하지 않는다.
"""
