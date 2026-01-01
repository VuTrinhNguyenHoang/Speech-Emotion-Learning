from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

EMO_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

SPEAKER_RE = re.compile(r"{\d{3,4}}")
EMO_RE = re.compile(r"_(ANG|DIS|FEA|HAP|NEU|SAD)_")

@dataclass(frozen=True)
class CremaItem:
    path: str
    speaker: str
    emotion: str

def parse_speaker_id(filename: str) -> str:
    m = SPEAKER_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse speaker id from: {filename}")
    return m.group(1)

def parse_emotion(filename: str) -> str:
    m = EMO_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse emotion from: {filename}")
    return EMO_MAP[m.group(1)]

def scan_crema(crema_wav_dir: Path) -> List[CremaItem]:
    wavs = sorted(crema_wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {crema_wav_dir}")
    items: list[CremaItem] = []
    for p in wavs:
        fn = p.name
        speaker = parse_speaker_id(fn)
        emotion = parse_emotion(fn)
        items.append(CremaItem(path=str(p), speaker=str(speaker), emotion=emotion))
    return items
