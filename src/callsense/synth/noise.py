"""Speech-to-text style noise, added in code after a transcript is written. Free and deterministic.

Real transcripts come from speech recognition: homophones get swapped, punctuation goes missing,
words are repeated, and people say "um". Lines that contain planted personal data are left alone,
so the redaction ground truth stays exact.
"""

import random
import re

NOISY_CALL_SHARE = 0.5
LINE_SHARE = 0.3  # share of lines touched in a noisy call

_HOMOPHONES = {
    "their": "there", "there": "their", "your": "you're", "to": "too", "its": "it's", "it's": "its",
    "then": "than", "than": "then", "know": "no", "right": "write", "by": "buy", "for": "four",
}
_FILLERS = ("um,", "uh,", "so,", "like,", "you know,")


def _swap_homophone(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    spots = [i for i, w in enumerate(words) if w.lower() in _HOMOPHONES]
    if spots:
        i = rng.choice(spots)
        words[i] = _HOMOPHONES[words[i].lower()]
    return " ".join(words)


def _drop_punctuation(text: str, rng: random.Random) -> str:
    return re.sub(r"[,.](?= )", lambda m: "" if rng.random() < 0.6 else m.group(0), text)


def _repeat_word(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    if len(words) > 3:
        i = rng.randrange(1, len(words) - 1)
        words.insert(i, words[i])
    return " ".join(words)


def _add_filler(text: str, rng: random.Random) -> str:
    return f"{rng.choice(_FILLERS)} {text[0].lower()}{text[1:]}" if text else text


_EDITS = (_swap_homophone, _drop_punctuation, _repeat_word, _add_filler)


def add_noise(transcript: str, call_id: str, protected: list[str]) -> tuple[str, bool]:
    """Return the transcript with speech-to-text noise, and whether any was added.

    The same call_id always gets the same noise. Lines containing any protected string are unchanged.
    """
    rng = random.Random(f"noise-{call_id}")
    if rng.random() >= NOISY_CALL_SHARE:
        return transcript, False

    out, changed = [], False
    for line in transcript.split("\n"):
        speaker, sep, text = line.partition(": ")
        if sep and text and rng.random() < LINE_SHARE and not any(p in line for p in protected):
            new = rng.choice(_EDITS)(text, rng)
            changed = changed or new != text
            line = f"{speaker}: {new}"
        out.append(line)
    return "\n".join(out), changed
