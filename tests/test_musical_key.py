from __future__ import annotations

from discql import musical_key


def test_format_key_returns_none_when_key_missing():
    assert musical_key.format_key(None, None) is None


def test_format_key_standard_spells_out_and_capitalizes_scale():
    assert musical_key.format_key("F#", "minor") == "F# Minor"
    assert musical_key.format_key("C", "major") == "C Major"


def test_format_key_standard_keeps_stored_spelling_as_is():
    # e.g. an unusual-but-valid enharmonic spelling like "E#" - the point is
    # not to silently normalize it to "F", just to display it cleanly.
    assert musical_key.format_key("E#", "minor") == "E# Minor"


def test_format_key_camelot_matches_known_pairs():
    assert musical_key.format_key("C", "major", "camelot") == "8B"
    assert musical_key.format_key("A", "minor", "camelot") == "8A"
    assert musical_key.format_key("F#", "minor", "camelot") == "11A"
    assert musical_key.format_key("D#", "major", "camelot") == "5B"


def test_format_key_camelot_resolves_enharmonic_aliases():
    # Eb minor and D# minor are the same pitch class - both map to 2A.
    assert musical_key.format_key("Eb", "minor", "camelot") == "2A"
    assert musical_key.format_key("D#", "minor", "camelot") == "2A"
    # E# is enharmonic to F.
    assert musical_key.format_key("E#", "minor", "camelot") == musical_key.format_key("F", "minor", "camelot")


def test_format_key_compact_abbreviates_scale_for_tight_layouts():
    assert musical_key.format_key("F#", "minor", compact=True) == "F# min"
    assert musical_key.format_key("C", "major", compact=True) == "C maj"


def test_format_key_camelot_falls_back_to_raw_for_unrecognized_key():
    assert musical_key.format_key("not-a-key", "minor", "camelot") == "not-a-key"


def test_camelot_wheel_is_internally_consistent():
    # Every (major, minor) pair sharing a key signature must share a number,
    # and every number 1-12 must be used exactly once per letter.
    majors = {code for (index, is_minor), code in musical_key._CAMELOT.items() if not is_minor}
    minors = {code for (index, is_minor), code in musical_key._CAMELOT.items() if is_minor}
    assert majors == {f"{n}B" for n in range(1, 13)}
    assert minors == {f"{n}A" for n in range(1, 13)}
