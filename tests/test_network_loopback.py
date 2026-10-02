"""서버(AudioServer + main.serve_network)와 클라이언트를 127.0.0.1에서 함께 띄우는 통합 테스트.

기본 실행 테스트는 레벨만 보고 판정하는 가짜 분류기를 쓴다(모델 불필요). CED와 ESC-50이 필요한
동등성 테스트는 slow 마커로 분리한다. 서버는 127.0.0.1에 포트 0(자동 할당)으로 bind한다.
0.0.0.0이면 Windows 방화벽 팝업이 뜰 수 있기 때문이다.
"""

from __future__ import annotations

import logging
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import pytest
from scipy.io import wavfile

import config
from capture import FileSource
from decision import LOCAL_TIMEZONE, DecisionConfig, NoiseDecisionEngine
from models import Category, Frame
from pi_client.capture import FileStream
from pi_client.main import PiClient
from protocol import (
    PROTOCOL_VERSION,
    AudioChunk,
    ConnectionClosed,
    Hello,
    HelloAck,
    Message,
    MessageType,
    MicInfo,
    decode_hello_ack,
    decode_status,
    encode_alert,
    encode_audio,
    encode_hello,
    encode_hello_ack,
    read_message,
    send_message,
)
from server_net import AudioServer, ClientSession

if TYPE_CHECKING:
    from classifier import ClassificationResult
    from main import RunResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOOPBACK_HOST = "127.0.0.1"
DAY_START = datetime(2026, 10, 2, 14, 0, tzinfo=LOCAL_TIMEZONE).timestamp()
SAMPLE_RATE = 48000
CHUNK_SAMPLES = 4800
CHUNK_SEC = CHUNK_SAMPLES / SAMPLE_RATE
LOUD_PEAK = 0.05  # 가짜 분류기: 창 최대 진폭이 이보다 크면 IMPACT로 본다
SERVER_TIMEOUT_SEC = 60.0


class LevelStubClassifier:
    """CED 대신 창의 최대 진폭으로 IMPACT 확률을 정하는 가짜 분류기."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(
        self, batch: dict[str, numpy.ndarray]
    ) -> dict[str, ClassificationResult]:
        from classifier import ClassificationResult

        self.calls += 1
        results = {}
        for name, window in batch.items():
            loud = float(numpy.max(numpy.abs(window))) > LOUD_PEAK
            probs = {category: 0.0 for category in Category}
            probs[Category.IMPACT] = 0.9 if loud else 0.0
            label = "Stub impact" if loud else "Stub silence"
            results[name] = ClassificationResult(numpy.zeros(1), probs, [(label, 0.9)])
        return results


def make_signal(
    duration_sec: float, bursts: list[float], amplitude: float
) -> numpy.ndarray:
    """무음 위에 0.3초짜리 1kHz 버스트를 bursts 시각(초)에 둔 int16 신호."""
    samples = numpy.zeros(round(duration_sec * SAMPLE_RATE))
    burst_time = numpy.arange(round(0.3 * SAMPLE_RATE)) / SAMPLE_RATE
    burst = amplitude * numpy.sin(2 * numpy.pi * 1000 * burst_time)
    for start_sec in bursts:
        start = round(start_sec * SAMPLE_RATE)
        samples[start : start + len(burst)] = burst
    return numpy.round(samples * 32767).astype(numpy.int16)


def chunks_of(signal: numpy.ndarray) -> list[numpy.ndarray]:
    return [
        signal[start : start + CHUNK_SAMPLES]
        for start in range(0, len(signal), CHUNK_SAMPLES)
    ]


@dataclass
class ServerRun:
    server: AudioServer
    thread: threading.Thread
    results: list


@pytest.fixture
def no_leftover_threads() -> Iterator[None]:
    yield
    deadline = time.monotonic() + config.THREAD_JOIN_TIMEOUT_SEC
    leftover = []
    while time.monotonic() < deadline:
        leftover = [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("noise-guard") and thread.is_alive()
        ]
        if not leftover:
            return
        time.sleep(0.05)
    pytest.fail(f"테스트 후 남은 스레드: {leftover}")


@pytest.fixture
def server(no_leftover_threads: None) -> Iterator[AudioServer]:
    audio_server = AudioServer(LOOPBACK_HOST, 0, start_time=DAY_START)
    audio_server.start()
    yield audio_server
    audio_server.close()


def start_serving(
    server: AudioServer,
    classifier: object,
    engine: NoiseDecisionEngine,
    sessions: int = 1,
) -> ServerRun:
    from main import serve_network

    results: list = []
    thread = threading.Thread(
        target=lambda: results.extend(
            serve_network(server, classifier, engine, None, None, max_sessions=sessions)
        ),
        name="test-serve",
    )
    thread.start()
    return ServerRun(server, thread, results)


def finish(run: ServerRun) -> list:
    run.thread.join(SERVER_TIMEOUT_SEC)
    assert not run.thread.is_alive(), "서버 처리가 끝나지 않았습니다"
    return run.results


class RawClient:
    """타이밍을 테스트가 직접 정하는 최소 클라이언트 (protocol만 사용)."""

    def __init__(self, port: int, rooms: list[str]) -> None:
        self.connection = socket.create_connection((LOOPBACK_HOST, port), timeout=10)
        mics = tuple(
            MicInfo(index, room, SAMPLE_RATE) for index, room in enumerate(rooms)
        )
        send_message(
            self.connection,
            MessageType.HELLO,
            encode_hello(Hello(PROTOCOL_VERSION, "raw", mics, CHUNK_SAMPLES)),
        )
        self.ack = decode_hello_ack(read_message(self.connection).body)

    def send_chunk(self, mic_index: int, seq: int, samples: numpy.ndarray) -> None:
        body = encode_audio(AudioChunk(mic_index, seq, 0, samples))
        send_message(self.connection, MessageType.AUDIO, body)

    def finish(self) -> list[Message]:
        """송신을 닫고 서버가 닫을 때까지 읽는다. 받은 메시지(ALERT, STATUS 등)를 반환한다."""
        self.connection.shutdown(socket.SHUT_WR)
        received = []
        try:
            while True:
                received.append(read_message(self.connection))
        except (ConnectionClosed, OSError):
            pass
        self.connection.close()
        return received


def send_schedule(
    client: RawClient,
    signals: list[numpy.ndarray],
    delay_for: Callable[[int, int], float],
) -> None:
    """청크 (mic, seq)를 '실시간 끝 시각 + delay_for(mic, seq)'에 보낸다."""
    start = time.monotonic()
    events = sorted(
        ((seq + 1) * CHUNK_SEC + delay_for(mic, seq), mic, seq, chunk)
        for mic, signal in enumerate(signals)
        for seq, chunk in enumerate(chunks_of(signal))
    )
    for due, mic, seq, chunk in events:
        time.sleep(max(0.0, start + due - time.monotonic()))
        client.send_chunk(mic, seq, chunk)


def run_raw(
    server: AudioServer,
    rooms: list[str],
    signals: list[numpy.ndarray],
    delay_for: Callable[[int, int], float] | None,
    skip_seqs: set[int] = frozenset(),
) -> tuple[RunResult, ClientSession, list[Message]]:
    """delay_for가 None이면 지연 없이 한꺼번에 보낸다. 서버가 보낸 메시지도 함께 반환한다."""
    engine = NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True))
    run = start_serving(server, LevelStubClassifier(), engine)
    client = RawClient(server.address[1], rooms)
    assert client.ack.accepted, client.ack.reasons
    if delay_for is None:
        for mic, signal in enumerate(signals):
            for seq, chunk in enumerate(chunks_of(signal)):
                if seq not in skip_seqs:
                    client.send_chunk(mic, seq, chunk)
    else:
        send_schedule(client, signals, delay_for)
    received = client.finish()
    session_result = finish(run)[0]
    return session_result.result, session_result.session, received


def frame_rows(frames: list[Frame]) -> list[tuple]:
    return [
        (
            round(frame.timestamp - frames[0].timestamp, 6),
            frame.mic_name,
            round(frame.leq_db, 6),
            round(frame.lmax_db, 6),
            frame.top_label,
        )
        for frame in frames
    ]


# --- 기본 동작 ---


def test_frame_timestamps_follow_audio_samples(server: AudioServer) -> None:
    signal = make_signal(5.0, bursts=[2.5], amplitude=0.5)
    result, session, _ = run_raw(server, ["거실"], [signal], delay_for=None)
    offsets = [frame.timestamp - session.stream_start_time for frame in result.frames]
    # 분류 창(2초)이 찬 tick 2부터 5까지, 오디오 1초마다 정확히 1프레임.
    assert offsets == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert abs(session.stream_start_time - DAY_START) < 5.0


def test_skipped_seq_is_filled_with_silence_without_shifting_time(
    server: AudioServer,
) -> None:
    signal = make_signal(5.0, bursts=[3.5], amplitude=0.5)
    result, session, _ = run_raw(
        server, ["거실"], [signal], delay_for=None, skip_seqs={10, 11, 12}
    )
    assert session.stats()["거실"].gap_filled_chunks == 3
    offsets = [frame.timestamp - session.stream_start_time for frame in result.frames]
    assert offsets == pytest.approx([2.0, 3.0, 4.0, 5.0])
    # 버스트(3.5초)는 시간축이 밀리지 않았다면 tick 4(3~4초)에 잡힌다.
    loud_offsets = [
        offset
        for offset, frame in zip(offsets, result.frames, strict=True)
        if frame.top_label == "Stub impact" and frame.lmax_db > 0
    ]
    assert loud_offsets[0] == pytest.approx(4.0)


def test_one_mic_delayed_is_excluded_and_others_continue(server: AudioServer) -> None:
    quiet_room = make_signal(5.0, bursts=[2.2], amplitude=0.2)
    loud_room = make_signal(5.0, bursts=[2.2], amplitude=0.9)
    started = time.monotonic()
    result, session, received = run_raw(
        server,
        ["거실", "안방"],
        [quiet_room, loud_room],
        delay_for=lambda mic, seq: 3.0 if mic == 1 else 0.0,
    )
    elapsed = time.monotonic() - started
    stats = session.stats()
    logging.getLogger(__name__).info("한 마이크 3초 지연: %.1f초, %s", elapsed, stats)
    # 안방(1번)은 늘 2초 마감을 넘기므로 모든 tick에서 빠지고, 거실 프레임은 정상으로 나온다.
    assert len(result.frames) == 4
    assert {frame.mic_name for frame in result.frames} == {"거실"}
    assert stats["안방"].stalled_ticks == 5
    assert stats["안방"].late_discarded_samples == 5 * SAMPLE_RATE
    assert stats["거실"].stalled_ticks == 0
    # 3 tick 연속으로 빠지면 Pi에 STATUS를 한 번 보낸다(상태가 바뀔 때만).
    statuses = [
        decode_status(message.body)
        for message in received
        if message.message_type == MessageType.STATUS
    ]
    assert [status.missing_mics for status in statuses] == [("안방",)]
    assert (statuses[0].active_mics, statuses[0].total_mics) == (1, 2)
    # tick 1은 분류 창이 차기 전이라 집계하지 않는다. 안방은 tick 2~5에서 빠졌다.
    assert result.missing_ticks == {"거실": 0, "안방": 4}


def test_all_mics_delayed_catch_up_without_loss(server: AudioServer) -> None:
    signals = [
        make_signal(6.0, bursts=[1.2, 4.2], amplitude=0.3),
        make_signal(6.0, bursts=[2.2, 4.6], amplitude=0.6),
    ]
    baseline, _, _ = run_raw(server, ["거실", "안방"], signals, delay_for=None)

    # 2초 지점부터 모든 마이크가 3초 멈췄다가, 밀린 청크를 한꺼번에 보내고 실시간으로 이어간다.
    def pause_after_two_seconds(mic: int, seq: int) -> float:
        chunk_end = (seq + 1) * CHUNK_SEC
        return max(0.0, 5.0 - chunk_end) if chunk_end > 2.0 else 0.0

    delayed, session, _ = run_raw(
        server, ["거실", "안방"], signals, delay_for=pause_after_two_seconds
    )
    stats = session.stats()
    assert all(room_stats.stalled_ticks == 0 for room_stats in stats.values())
    assert all(room_stats.late_discarded_samples == 0 for room_stats in stats.values())
    assert frame_rows(delayed.frames) == frame_rows(baseline.frames)
    assert [(alert.rule, alert.mic_name) for alert in delayed.alerts] == [
        (alert.rule, alert.mic_name) for alert in baseline.alerts
    ]
    logging.getLogger(__name__).info("따라잡기 lag(ms): %s", delayed.lag_ms)


# --- 오류 정책 ---


def test_unknown_mic_index_disconnects(server: AudioServer) -> None:
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
    )
    client = RawClient(server.address[1], ["거실"])
    client.send_chunk(7, 0, numpy.zeros(CHUNK_SAMPLES, numpy.int16))
    client.finish()
    session = finish(run)[0].session
    assert "mic_index 7" in session.close_reason


def test_broken_framing_disconnects(server: AudioServer) -> None:
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
    )
    client = RawClient(server.address[1], ["거실"])
    client.connection.sendall((0).to_bytes(4, "big"))
    client.finish()
    session = finish(run)[0].session
    assert "FramingError" in session.close_reason


def test_invalid_hello_is_rejected(server: AudioServer) -> None:
    connection = socket.create_connection(
        (LOOPBACK_HOST, server.address[1]), timeout=10
    )
    hello = Hello(PROTOCOL_VERSION, "bad", (MicInfo(0, "거실", 22050),), CHUNK_SAMPLES)
    send_message(connection, MessageType.HELLO, encode_hello(hello))
    ack = decode_hello_ack(read_message(connection).body)
    with pytest.raises(ConnectionClosed):
        read_message(connection)
    connection.close()
    assert not ack.accepted
    assert any("sample_rate" in reason for reason in ack.reasons)


def test_second_client_is_rejected(server: AudioServer) -> None:
    first = RawClient(server.address[1], ["거실"])
    session = server.next_session(timeout=5.0)
    assert first.ack.accepted and session is not None
    second = RawClient(server.address[1], ["안방"])
    assert not second.ack.accepted
    assert any("이미" in reason for reason in second.ack.reasons)
    second.connection.close()
    first.connection.close()
    session.close()
    session.join()


def run_pi_client_against_session(
    server: AudioServer, wav_path: Path, on_session: Callable[[ClientSession], None]
) -> tuple[PiClient, RunResult]:
    """PiClient(FileStream, 실시간)를 띄우고, 서버 세션이 생기면 on_session을 부른다."""
    from main import run_sources

    client = PiClient(
        LOOPBACK_HOST,
        server.address[1],
        "pi-test",
        FileStream({"거실": wav_path}, CHUNK_SAMPLES, fast=False),
        CHUNK_SAMPLES,
    )
    client_thread = threading.Thread(target=client.run, name="test-pi-client")
    client_thread.start()
    session = server.next_session(timeout=10.0)
    assert session is not None
    holder: list = []
    runner = threading.Thread(
        target=lambda: holder.append(
            run_sources(
                session.sources,
                LevelStubClassifier(),
                NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
                {
                    "start_timestamp": session.stream_start_time,
                    "realtime_pacing": False,
                    "stop_when_exhausted": True,
                    "duration_sec": None,
                },
                None,
                None,
            )
        ),
        name="test-run-sources",
    )
    runner.start()
    on_session(session)
    # serve_network와 같은 순서: 처리가 끝나면 세션을 닫고, 클라이언트는 그 종료를 보고 끝난다.
    runner.join(SERVER_TIMEOUT_SEC)
    session.close()
    session.join()
    client_thread.join(SERVER_TIMEOUT_SEC)
    assert not client_thread.is_alive() and not runner.is_alive()
    return client, holder[0]


def valid_alert_payload() -> dict[str, object]:
    return {
        "level": "caution",
        "rules": ["R1"],
        "room": "거실",
        "category": "impact",
        "label": "Walk, footsteps",
        "label_ko": "발소리",
        "peak_db": 60.0,
        "count": 1,
        "room_counts": {"거실": 1},
        "message": "test",
        "timestamp": DAY_START,
        "source_mic_index": 0,
        "source_seq": 0,
    }


def write_test_wav(tmp_path: Path, duration_sec: float) -> Path:
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, make_signal(duration_sec, [1.5], 0.5))
    return path


def test_client_ignores_bad_alert_payload_and_keeps_sending(
    server: AudioServer, tmp_path: Path
) -> None:
    wav_path = write_test_wav(tmp_path, 3.0)

    def send_alerts(session: ClientSession) -> None:
        session.send(MessageType.ALERT, b"{broken json")
        session.send(MessageType.ALERT, encode_alert(valid_alert_payload()))

    client, result = run_pi_client_against_session(server, wav_path, send_alerts)
    assert client.ignored_payloads == 1
    assert len(client.received_alerts) == 1
    assert client.sent_chunks == 30  # 3초 / 100ms, 끝까지 보냄
    assert len(result.frames) == 2


def test_client_disconnects_on_framing_error(
    server: AudioServer, tmp_path: Path
) -> None:
    wav_path = write_test_wav(tmp_path, 5.0)

    def send_garbage(session: ClientSession) -> None:
        time.sleep(1.0)
        session.connection.sendall((0).to_bytes(4, "big"))

    client, _ = run_pi_client_against_session(server, wav_path, send_garbage)
    assert client.sent_chunks < 50  # 5초 분량을 다 보내기 전에 끊었다


def test_pi_client_does_not_import_server_or_heavy_modules() -> None:
    forbidden = [
        "torch",
        "transformers",
        "scipy",
        "config",
        "capture",
        "classifier",
        "main",
        "server_net",
        "decision",
        "level",
        "label_map",
        "fusion",
        "alert",
        "models",
    ]
    code = (
        "import sys, pi_client.main, pi_client.capture, pi_client.client_config;"
        f"print([m for m in {forbidden!r} if m in sys.modules])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "[]"


# --- 동등성 (CED + ESC-50 필요) ---


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["S1_footsteps_x3", "S5_two_rooms"])
def test_network_path_matches_direct_file_path(
    server: AudioServer, tmp_path: Path, scenario: str
) -> None:
    from classifier import CedClassifier
    from main import run_sources
    from tools.make_scenario import SCENARIOS, write_scenario

    esc50_dir = PROJECT_ROOT / "data" / "esc50" / "audio"
    if not esc50_dir.is_dir():
        pytest.skip(f"ESC-50 클립이 없습니다: {esc50_dir}")
    write_scenario(
        scenario, SCENARIOS[scenario], esc50_dir, tmp_path, extra_gain_db=0.0
    )
    files = {room: tmp_path / f"{scenario}_{room}.wav" for room in SCENARIOS[scenario]}
    classifier = CedClassifier()

    direct = run_sources(
        [FileSource(room, path, end_behavior="stop") for room, path in files.items()],
        classifier,
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        {
            "start_timestamp": DAY_START,
            "realtime_pacing": False,
            "stop_when_exhausted": True,
            "duration_sec": None,
        },
        None,
        None,
    )

    run = start_serving(
        server,
        classifier,
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
    )
    client = PiClient(
        LOOPBACK_HOST,
        server.address[1],
        "pi-test",
        FileStream(files, CHUNK_SAMPLES, fast=True),
        CHUNK_SAMPLES,
    )
    client.run()
    network = finish(run)[0].result

    def alert_summary(result: RunResult) -> Counter:
        return Counter(
            (alert.rule, alert.mic_name, alert.count) for alert in result.alerts
        )

    assert alert_summary(network) == alert_summary(direct)
    assert [frame.mic_name for frame in network.frames] == [
        frame.mic_name for frame in direct.frames
    ]


class TimedRecordingOutput:
    """출력 시각과 그때까지 보낸 청크 수를 기록한다. 표시 유지 중 전송이 계속되는지 보기 위해서다."""

    def __init__(self) -> None:
        self.client: PiClient | None = None
        self.events: list[tuple[str, float, int]] = []

    def _record(self, kind: str) -> None:
        sent = self.client.sent_chunks if self.client else 0
        self.events.append((kind, time.monotonic(), sent))

    def show_caution(self, alert: dict) -> None:
        self._record("caution")

    def show_warning(self, alert: dict) -> None:
        self._record("warning")

    def show_status(self, status: object) -> None:
        self._record("status")

    def clear(self) -> None:
        self._record("clear")


def test_pi_client_shows_alerts_keeps_sending_and_measures_latency(
    server: AudioServer, tmp_path: Path
) -> None:
    # 4초 간격 버스트 3개: t≈2 warning(R1+R3), t≈6 caution(R1, R3는 쿨다운), t≈10 warning(R1+R2).
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, make_signal(11.0, [1.5, 5.5, 9.5], 0.5))
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
    )
    output = TimedRecordingOutput()
    client = PiClient(
        LOOPBACK_HOST,
        server.address[1],
        "pi-test",
        FileStream({"거실": path}, CHUNK_SAMPLES, fast=False),
        CHUNK_SAMPLES,
        output=output,
        display_hold_sec=2.0,
    )
    output.client = client
    client.run()
    finish(run)

    kinds = [kind for kind, _, _ in output.events]
    assert kinds[:3] == ["warning", "clear", "caution"]
    assert kinds.count("warning") == 2
    levels = [alert["level"] for alert in client.received_alerts]
    assert levels == ["warning", "caution", "warning"]
    assert client.received_alerts[2]["rules"] == ["R1", "R2"]

    # 첫 warning 표시부터 clear까지 2초 동안에도 100ms 청크가 계속 나갔다.
    (_, shown_at, sent_at_show), (_, cleared_at, sent_at_clear) = output.events[:2]
    assert cleared_at - shown_at >= 1.9
    assert sent_at_clear - sent_at_show >= 15

    assert client.unmeasured_alerts == 0
    assert len(client.latencies_ms) == 3
    assert all(
        0.0 <= latency < 3000.0 for latency in client.latencies_ms
    ), client.latencies_ms


def test_no_missing_mics_reported_at_startup(server: AudioServer) -> None:
    signals = [make_signal(5.0, [], 0.0), make_signal(5.0, [], 0.0)]
    result, _, received = run_raw(server, ["거실", "안방"], signals, delay_for=None)
    assert result.missing_ticks == {"거실": 0, "안방": 0}
    assert not [m for m in received if m.message_type == MessageType.STATUS]


# --- N4: 마이크 멈춤(--stall), 재접속, 타임아웃 ---


def test_stalled_mic_shows_status_then_recovers(
    server: AudioServer, tmp_path: Path
) -> None:
    paths = {}
    for room in ("거실", "안방"):
        paths[room] = tmp_path / f"{room}.wav"
        wavfile.write(paths[room], SAMPLE_RATE, make_signal(11.0, [], 0.0))
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
    )
    client = PiClient(
        LOOPBACK_HOST,
        server.address[1],
        "pi-test",
        FileStream(paths, CHUNK_SAMPLES, fast=False, stalls={"안방": [(2.0, 6.0)]}),
        CHUNK_SAMPLES,
        output=TimedRecordingOutput(),
    )
    client.run()
    result = finish(run)[0].result
    # 6초 멈춤 중 마감(2초) 안에 무음 채움이 도착하지 못한 tick만 빠짐으로 센다(약 6 - 2 = 4개).
    assert [status.missing_mics for status in client.received_statuses] == [
        ("안방",),
        (),
    ]
    assert result.missing_ticks["거실"] == 0
    assert 3 <= result.missing_ticks["안방"] <= 5


def test_client_reconnects_after_server_drops_session(
    server: AudioServer, tmp_path: Path
) -> None:
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, make_signal(6.0, [], 0.0))
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        sessions=2,
    )
    client = PiClient(
        LOOPBACK_HOST,
        server.address[1],
        "pi-test",
        FileStream({"거실": path}, CHUNK_SAMPLES, fast=False),
        CHUNK_SAMPLES,
        output=TimedRecordingOutput(),
        reconnect=True,
        reconnect_initial_sec=0.2,
    )

    def drop_first_session() -> None:
        deadline = time.monotonic() + 10.0
        while server.active_session is None and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(2.0)
        server.active_session.close("테스트: 서버가 세션을 끊음")

    dropper = threading.Thread(target=drop_first_session, name="test-dropper")
    dropper.start()
    client.run()
    dropper.join()
    session_results = finish(run)
    assert client.connections == 2
    assert "서버가 연결을 닫음" in client.lost_reasons[0]
    assert len(session_results) == 2
    assert session_results[0].session.close_reason == "테스트: 서버가 세션을 끊음"
    assert session_results[1].result.frames, "재접속한 세션도 프레임을 만들어야 합니다"


def test_decision_state_survives_reconnect(server: AudioServer) -> None:
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        sessions=2,
    )
    # 첫 연결: 충격 이벤트 2개(R1 1, 2회). 두 번째 연결: 1개 → 카운트 3회가 되어 R2 경고.
    for signal in (
        make_signal(10.0, [1.5, 6.0], 0.5),
        make_signal(5.0, [1.5], 0.5),
    ):
        client = RawClient(server.address[1], ["거실"])
        assert client.ack.accepted, client.ack.reasons
        for seq, chunk in enumerate(chunks_of(signal)):
            client.send_chunk(0, seq, chunk)
        client.finish()
    first, second = finish(run)
    assert [a.count for a in first.result.alerts if a.rule == "R1"] == [1, 2]
    second_r1 = [a for a in second.result.alerts if a.rule == "R1"]
    second_r2 = [a for a in second.result.alerts if a.rule == "R2"]
    assert [alert.count for alert in second_r1] == [3]
    assert len(second_r2) == 1 and second_r2[0].room_counts == {"거실": 3}
    assert second.result.frames[0].timestamp > first.result.frames[-1].timestamp


def test_server_survives_client_reset_and_accepts_next(server: AudioServer) -> None:
    run = start_serving(
        server,
        LevelStubClassifier(),
        NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        sessions=2,
    )
    first = RawClient(server.address[1], ["거실"])
    for seq, chunk in enumerate(chunks_of(make_signal(3.0, [], 0.0))):
        first.send_chunk(0, seq, chunk)
    # 프로세스가 강제 종료된 것처럼 FIN 대신 RST로 끊는다.
    first.connection.setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    first.connection.close()
    deadline = time.monotonic() + 10.0
    while server.active_session is not None and server.active_session.connected:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    second = RawClient(server.address[1], ["거실"])
    assert second.ack.accepted, second.ack.reasons
    for seq, chunk in enumerate(chunks_of(make_signal(3.0, [], 0.0))):
        second.send_chunk(0, seq, chunk)
    second.finish()
    session_results = finish(run)
    assert "소켓 오류" in session_results[0].session.close_reason
    assert len(session_results[1].result.frames) == 2


def test_server_closes_client_that_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, no_leftover_threads: None
) -> None:
    monkeypatch.setattr(config, "PEER_TIMEOUT_SEC", 0.9)
    monkeypatch.setattr(config, "PING_INTERVAL_SEC", 0.3)
    quiet_server = AudioServer(LOOPBACK_HOST, 0, start_time=DAY_START)
    quiet_server.start()
    try:
        run = start_serving(
            quiet_server,
            LevelStubClassifier(),
            NoiseDecisionEngine(DecisionConfig.from_config(demo_mode=True)),
        )
        client = RawClient(quiet_server.address[1], ["거실"])
        received = []
        try:
            while True:
                received.append(read_message(client.connection).message_type)
        except (ConnectionClosed, OSError):
            pass
        client.connection.close()
        session = finish(run)[0].session
    finally:
        quiet_server.close()
    assert "수신 없음" in session.close_reason
    assert MessageType.PING in received  # 서버는 그동안 PING을 보냈다


def test_client_detects_server_that_goes_silent(
    tmp_path: Path, no_leftover_threads: None
) -> None:
    listener = socket.create_server((LOOPBACK_HOST, 0))
    accepted: list[socket.socket] = []

    def silent_server() -> None:
        connection, _ = listener.accept()
        accepted.append(connection)
        read_message(connection)  # HELLO
        send_message(
            connection,
            MessageType.HELLO_ACK,
            encode_hello_ack(HelloAck(True, (), time.time())),
        )
        # 이후로는 아무것도 보내지 않는다(PING도 없음).

    server_thread = threading.Thread(target=silent_server, name="test-silent-server")
    server_thread.start()
    path = tmp_path / "room.wav"
    wavfile.write(path, SAMPLE_RATE, make_signal(5.0, [], 0.0))
    client = PiClient(
        LOOPBACK_HOST,
        listener.getsockname()[1],
        "pi-test",
        FileStream({"거실": path}, CHUNK_SAMPLES, fast=False),
        CHUNK_SAMPLES,
        output=TimedRecordingOutput(),
        ping_interval_sec=0.3,
        peer_timeout_sec=0.9,
    )
    started = time.monotonic()
    exit_code = client.run()
    elapsed = time.monotonic() - started
    server_thread.join(5.0)
    for connection in accepted:
        connection.close()
    listener.close()
    assert exit_code != 0
    assert "수신 없음" in client.lost_reasons[0]
    assert elapsed < 3.0
