"""AudioSet 라벨(CED id2label 문자열) → Category 매핑 (순수 로직).

라벨 문자열은 tools/dump_labels.py 로 저장한 labels_dump.txt 의 값을 그대로 쓴다.
"""

from typing import Literal

import numpy

from models import Category

AirborneScope = Literal["extended", "legal"]

IMPACT_LABELS: frozenset[str] = frozenset(
    {
        "Run",
        "Shuffle",
        "Walk, footsteps",
        "Door",
        "Slam",
        "Knock",
        "Tap",
        "Hammer",
        "Thump, thud",
        "Thunk",
        "Basketball bounce",
        "Bang",
        "Whack, thwack",
        "Smash, crash",
        "Bouncing",
    }
)

# 법적 공기전달소음(텔레비전·음향기기 등)에 해당하는 라벨. AudioSet Music 하위 전체(137–282) 포함.
AIRBORNE_LEGAL_LABELS: frozenset[str] = frozenset(
    {
        "Television",
        "Radio",
        "Music",
        "Musical instrument",
        "Plucked string instrument",
        "Guitar",
        "Electric guitar",
        "Bass guitar",
        "Acoustic guitar",
        "Steel guitar, slide guitar",
        "Tapping (guitar technique)",
        "Strum",
        "Banjo",
        "Sitar",
        "Mandolin",
        "Zither",
        "Ukulele",
        "Keyboard (musical)",
        "Piano",
        "Electric piano",
        "Organ",
        "Electronic organ",
        "Hammond organ",
        "Synthesizer",
        "Sampler",
        "Harpsichord",
        "Percussion",
        "Drum kit",
        "Drum machine",
        "Drum",
        "Snare drum",
        "Rimshot",
        "Drum roll",
        "Bass drum",
        "Timpani",
        "Tabla",
        "Cymbal",
        "Hi-hat",
        "Wood block",
        "Tambourine",
        "Rattle (instrument)",
        "Maraca",
        "Gong",
        "Tubular bells",
        "Mallet percussion",
        "Marimba, xylophone",
        "Glockenspiel",
        "Vibraphone",
        "Steelpan",
        "Orchestra",
        "Brass instrument",
        "French horn",
        "Trumpet",
        "Trombone",
        "Bowed string instrument",
        "String section",
        "Violin, fiddle",
        "Pizzicato",
        "Cello",
        "Double bass",
        "Wind instrument, woodwind instrument",
        "Flute",
        "Saxophone",
        "Clarinet",
        "Harp",
        "Bell",
        "Church bell",
        "Jingle bell",
        "Bicycle bell",
        "Tuning fork",
        "Chime",
        "Wind chime",
        "Change ringing (campanology)",
        "Harmonica",
        "Accordion",
        "Bagpipes",
        "Didgeridoo",
        "Shofar",
        "Theremin",
        "Singing bowl",
        "Scratching (performance technique)",
        "Pop music",
        "Hip hop music",
        "Beatboxing",
        "Rock music",
        "Heavy metal",
        "Punk rock",
        "Grunge",
        "Progressive rock",
        "Rock and roll",
        "Psychedelic rock",
        "Rhythm and blues",
        "Soul music",
        "Reggae",
        "Country",
        "Swing music",
        "Bluegrass",
        "Funk",
        "Folk music",
        "Middle Eastern music",
        "Jazz",
        "Disco",
        "Classical music",
        "Opera",
        "Electronic music",
        "House music",
        "Techno",
        "Dubstep",
        "Drum and bass",
        "Electronica",
        "Electronic dance music",
        "Ambient music",
        "Trance music",
        "Music of Latin America",
        "Salsa music",
        "Flamenco",
        "Blues",
        "Music for children",
        "New-age music",
        "Vocal music",
        "A capella",
        "Music of Africa",
        "Afrobeat",
        "Christian music",
        "Gospel music",
        "Music of Asia",
        "Carnatic music",
        "Music of Bollywood",
        "Ska",
        "Traditional music",
        "Independent music",
        "Song",
        "Background music",
        "Theme music",
        "Jingle (music)",
        "Soundtrack music",
        "Lullaby",
        "Video game music",
        "Christmas music",
        "Dance music",
        "Wedding music",
        "Happy music",
        "Funny music",
        "Sad music",
        "Tender music",
        "Exciting music",
        "Angry music",
        "Scary music",
    }
)

# 법적 기준 밖이지만 이웃 배려 차원에서 공기전달로 보는 라벨 (scope="extended"일 때만 추가).
AIRBORNE_EXTENDED_LABELS: frozenset[str] = frozenset(
    {
        # 말소리·목소리 (0–37)
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
        "Speech synthesizer",
        "Shout",
        "Bellow",
        "Whoop",
        "Yell",
        "Battle cry",
        "Children shouting",
        "Screaming",
        "Whispering",
        "Laughter",
        "Baby laughter",
        "Giggle",
        "Snicker",
        "Belly laugh",
        "Chuckle, chortle",
        "Crying, sobbing",
        "Baby cry, infant cry",
        "Whimper",
        "Wail, moan",
        "Sigh",
        "Singing",
        "Choir",
        "Yodeling",
        "Chant",
        "Mantra",
        "Male singing",
        "Female singing",
        "Child singing",
        "Synthetic singing",
        "Rapping",
        "Humming",
        "Whistling",
        # 군중 (66–71)
        "Cheering",
        "Chatter",
        "Crowd",
        "Hubbub, speech noise, speech babble",
        "Children playing",
        # 개·고양이
        "Dog",
        "Bark",
        "Yip",
        "Howl",
        "Bow-wow",
        "Growling",
        "Whimper (dog)",
        "Cat",
        "Purr",
        "Meow",
        "Hiss",
        "Caterwaul",
        "Canidae, dogs, wolves",
        # 생활 기기
        "Vacuum cleaner",
        "Blender",
    }
)

EXCLUDED_LABELS: frozenset[str] = frozenset(
    {
        "Water",
        "Gurgling",
        "Stream",
        "Waterfall",
        "Water tap, faucet",
        "Sink (filling or washing)",
        "Bathtub (filling or washing)",
        "Toilet flush",
        "Liquid",
        "Splash, splatter",
        "Slosh",
        "Drip",
        "Pour",
        "Trickle, dribble",
        "Gush",
        "Fill (with liquid)",
        "Spray",
    }
)


def airborne_labels(scope: AirborneScope) -> frozenset[str]:
    """scope에 따라 AIRBORNE으로 볼 라벨 집합을 반환한다. extended는 legal을 포함한다."""
    if scope == "legal":
        return AIRBORNE_LEGAL_LABELS
    if scope == "extended":
        return AIRBORNE_LEGAL_LABELS | AIRBORNE_EXTENDED_LABELS
    raise ValueError(f"알 수 없는 AIRBORNE scope: {scope!r}")


def category_for_label(label: str, scope: AirborneScope) -> Category:
    """라벨 하나의 카테고리. 어디에도 없으면 OTHER."""
    if label in IMPACT_LABELS:
        return Category.IMPACT
    if label in airborne_labels(scope):
        return Category.AIRBORNE
    if label in EXCLUDED_LABELS:
        return Category.EXCLUDED
    return Category.OTHER


def build_category_indices(
    labels: list[str], scope: AirborneScope
) -> dict[Category, numpy.ndarray]:
    """id2label 순서의 라벨 목록 → 카테고리별 라벨 인덱스 배열. 추론 전 한 번만 만든다."""
    indices_by_category: dict[Category, list[int]] = {
        category: [] for category in Category
    }
    for index, label in enumerate(labels):
        indices_by_category[category_for_label(label, scope)].append(index)
    return {
        category: numpy.array(indices, dtype=numpy.int64)
        for category, indices in indices_by_category.items()
    }


def category_probabilities(
    probabilities: numpy.ndarray, category_indices: dict[Category, numpy.ndarray]
) -> dict[Category, float]:
    """카테고리별 확률 = 해당 카테고리 라벨 확률의 최댓값. 라벨이 없는 카테고리는 0."""
    return {
        category: float(probabilities[indices].max()) if len(indices) else 0.0
        for category, indices in category_indices.items()
    }
