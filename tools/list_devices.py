"""오디오 입력 장치 목록을 출력한다 (side effect: 오디오 장치 조회, 콘솔 출력).

config.MIC_DEVICES에 넣을 장치 인덱스/이름을 확인하는 용도.
"""

import sounddevice


def print_input_devices() -> None:
    """입력 채널이 있는 장치만 호스트 API와 함께 콘솔에 출력한다."""
    host_apis = sounddevice.query_hostapis()
    default_input_index = sounddevice.default.device[0]
    for index, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        marker = "*" if index == default_input_index else " "
        host_api_name = host_apis[device["hostapi"]]["name"]
        print(
            f"{marker}[{index:2d}] {device['name']} | {host_api_name} | "
            f"in={device['max_input_channels']} "
            f"default_sr={device['default_samplerate']:.0f}"
        )


if __name__ == "__main__":
    print_input_devices()
