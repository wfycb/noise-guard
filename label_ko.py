"""알림 표시용 한국어 라벨 (판정과 무관, label_map.py와 분리). 표에 없으면 영어 라벨을 그대로 쓴다."""

LABEL_KO: dict[str, str] = {
    "Walk, footsteps": "발소리",
    "Run": "뛰는 소리",
    "Shuffle": "발 끄는 소리",
    "Knock": "노크",
    "Tap": "두드리는 소리",
    "Thump, thud": "쿵 소리",
    "Bang": "쾅 소리",
    "Slam": "문 쾅",
    "Door": "문 소리",
    "Hammer": "망치질",
    "Smash, crash": "부딪히는 소리",
    "Basketball bounce": "공 튀기는 소리",
    "Speech": "말소리",
    "Television": "TV 소리",
    "Music": "음악",
    "Piano": "피아노",
    "Dog": "개",
    "Bark": "개 짖는 소리",
    "Baby cry, infant cry": "아기 울음",
    "Vacuum cleaner": "청소기",
}


def to_korean(label: str) -> str:
    return LABEL_KO.get(label, label)
