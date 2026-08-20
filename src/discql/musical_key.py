from __future__ import annotations

# Essentia's KeyExtractor returns a pitch class name (sharps only, e.g. "F#")
# plus a scale ("major"/"minor"). _ALIASES additionally covers enharmonic
# spellings that could show up (flats, or unusual-but-valid ones like "E#"/
# "B#") so a Camelot lookup still works regardless of which spelling is
# stored - the spelling itself is never changed/normalized for display.
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ALIASES = {
    "Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11,
    "B#": 0, "E#": 5,
}

# Camelot Wheel: (semitone index, is_minor) -> code. This is the standard
# mapping used by Mixed In Key/rekordbox, not a DiscQL invention - each
# major/minor pair sharing a key signature gets the same number, minor is
# "A", major is "B".
_CAMELOT: dict[tuple[int, bool], str] = {
    (8, True): "1A", (11, False): "1B",
    (3, True): "2A", (6, False): "2B",
    (10, True): "3A", (1, False): "3B",
    (5, True): "4A", (8, False): "4B",
    (0, True): "5A", (3, False): "5B",
    (7, True): "6A", (10, False): "6B",
    (2, True): "7A", (5, False): "7B",
    (9, True): "8A", (0, False): "8B",
    (4, True): "9A", (7, False): "9B",
    (11, True): "10A", (2, False): "10B",
    (6, True): "11A", (9, False): "11B",
    (1, True): "12A", (4, False): "12B",
}


def _semitone_index(key: str) -> int | None:
    key = key.strip()
    if key in _PITCH_CLASSES:
        return _PITCH_CLASSES.index(key)
    return _ALIASES.get(key)


def format_key(key: str | None, scale: str | None, notation: str = "standard", compact: bool = False) -> str | None:
    """Formats a raw Essentia key+scale pair (e.g. "F#"/"minor") for display.

    notation="standard": the stored spelling as-is (ASCII "#"/"b" accidental,
    same as PDF base fonts render - Unicode ♯/♭ glyphs aren't available
    without bundling a custom font), with the scale spelled out and
    capitalized (e.g. "F# Minor") - or, if compact=True (tight spaces like a
    sticker's track table), abbreviated instead (e.g. "F# min").
    notation="camelot": DJ Camelot Wheel notation (e.g. "11A", already short
    regardless of compact). Falls back to the standard spelling if the key
    isn't a recognized pitch class name.

    Returns None if key is missing (caller decides the placeholder).
    """
    if not key:
        return None
    is_minor = (scale or "").lower() == "minor"

    if notation == "camelot":
        index = _semitone_index(key)
        if index is None:
            return key
        return _CAMELOT.get((index, is_minor), key)

    if not scale:
        scale_label = ""
    elif compact:
        scale_label = "min" if is_minor else "maj"
    else:
        scale_label = "Minor" if is_minor else "Major"
    return f"{key} {scale_label}".strip()
