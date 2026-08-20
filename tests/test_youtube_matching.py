from __future__ import annotations

import json
import sqlite3

import pytest

from discql import db, youtube_matching as ym
from discql.discogs_api import ArtistData, ReleaseData, TrackData, VideoData
from discql.sync import upsert_release


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.migrate(connection)
    yield connection
    connection.close()


# --- parse_track_duration -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("8:49", 529),
        ("1:02:33", 3753),
        ("0:30", 30),
        (None, None),
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_track_duration(text, expected):
    assert ym.parse_track_duration(text) == expected


# --- denoise_video_title -----------------------------------------------------


def test_denoise_video_title_strips_artist_release_and_year():
    title = 'ARTIST NAME 1983 "Track Title Here" (German) ALBUM NAME #hashtag'
    result = ym.denoise_video_title(title, "Album Name", ["Artist Name"])
    assert "1983" not in result
    assert "ARTIST NAME" not in result.upper() or "Artist Name" not in result
    assert "ALBUM NAME" not in result.upper()
    assert "Track Title Here" in result


def test_denoise_video_title_is_case_insensitive():
    title = "artist name - some track"
    result = ym.denoise_video_title(title, "", ["Artist Name"])
    assert "artist name" not in result.lower()


def test_denoise_video_title_no_changes_returns_similar_string():
    title = "Completely Unrelated Text"
    result = ym.denoise_video_title(title, "Some Release", ["Some Artist"])
    assert "Completely Unrelated Text" == result


def test_denoise_video_title_strips_configured_extra_terms():
    title = "Some Track (Original Mix) [Label Rip]"
    result = ym.denoise_video_title(title, "", [], extra_terms=["Original Mix"])
    assert "Original Mix" not in result
    assert "Some Track" in result


def test_denoise_video_title_extra_terms_are_case_insensitive_whole_phrases():
    # "Original Mix" is one configured term (not split into "Original" and
    # "Mix" separately), and matches regardless of case in the video title.
    result = ym.denoise_video_title(
        "Some Track (ORIGINAL MIX)", "", [], extra_terms=["Original Mix"]
    )
    assert "original" not in result.lower()
    assert "mix" not in result.lower()
    assert "Some Track" in result


def test_denoise_video_title_without_extra_terms_leaves_them_in():
    title = "Some Track (Original Mix)"
    result = ym.denoise_video_title(title, "", [])
    assert "Original Mix" in result


# --- fuzzy_score ------------------------------------------------------------


def test_fuzzy_score_exact_title_and_duration_scores_high():
    score = ym.fuzzy_score("Rave-O-Lution", "Dance 2 Trance", 270, "Rave-O-Lution", 270)
    assert score > 0.9


def test_fuzzy_score_duration_mismatch_penalizes():
    close_duration = ym.fuzzy_score("Some Track", None, 300, "Some Track", 305)
    far_duration = ym.fuzzy_score("Some Track", None, 300, "Some Track", 900)
    assert close_duration > far_duration


def test_fuzzy_score_is_case_insensitive():
    same_case = ym.fuzzy_score("Rave-O-Lution", None, None, "Rave-O-Lution", None)
    different_case = ym.fuzzy_score("Rave-O-Lution", None, None, "RAVE-O-LUTION", None)
    assert different_case == same_case == 1.0


def test_fuzzy_score_unrelated_titles_score_low():
    score = ym.fuzzy_score("Rave-O-Lution", None, 270, "Completely Different Song Name", 9999)
    assert score < 0.3


def test_fuzzy_score_duration_deviance_over_threshold_zeroes_score_even_for_exact_title():
    # >20% duration deviance disqualifies the pair outright, regardless of
    # how well the titles match - no point running fuzzy title comparison
    # (let alone an LLM call) on a recording that's clearly not the same one.
    score = ym.fuzzy_score("Exact Same Title", None, 200, "Exact Same Title", 245)
    assert (245 - 200) / 200 > ym.DURATION_DEVIANCE_THRESHOLD
    assert score == 0.0


def test_fuzzy_score_duration_deviance_at_threshold_is_not_zeroed():
    # Right at (just under) the cutoff, the normal title+duration blend
    # still applies - the pre-filter isn't overly aggressive.
    score = ym.fuzzy_score("Exact Same Title", None, 200, "Exact Same Title", 235)
    assert (235 - 200) / 200 < ym.DURATION_DEVIANCE_THRESHOLD
    assert score > 0.9


def test_fuzzy_score_duration_deviance_filter_only_applies_when_both_durations_known():
    # Missing duration on either side means there's nothing to pre-filter on;
    # falls back to pure title score, unaffected by the other duration value.
    score = ym.fuzzy_score("Exact Same Title", None, None, "Exact Same Title", 9999)
    assert score > 0.9


# --- resolve_release: pure heuristics (no LLM) ------------------------------


def _track(id_, position, title, duration_seconds=None, artist=None):
    return ym.TrackInput(id=id_, position=position, title=title, artist=artist, duration_seconds=duration_seconds)


def _video(id_, title, duration=None, chapters=None):
    return ym.VideoInput(id=id_, title=title, description=None, duration=duration, chapters=chapters or [])


def _unreachable_ollama(prompt: str) -> str:
    raise AssertionError("Ollama should not have been called")


def test_resolve_release_confident_match_skips_llm():
    tracks = [_track(1, "A1", "Rave-O-Lution", 270)]
    videos = [_video(100, "Rave-O-Lution", 270)]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert len(results) == 1
    assert results[0].track_id == 1
    assert results[0].video_type == "full_track"
    assert results[0].is_selected is True
    assert results[0].llm_confidence is None


def test_resolve_release_force_llm_bypasses_confident_shortcut():
    # Without force_llm, this scores well above CONFIDENT_THRESHOLD and never
    # reaches the LLM. With force_llm=True, it must go through the LLM anyway.
    tracks = [_track(1, "A1", "Rave-O-Lution", 270)]
    videos = [_video(100, "Rave-O-Lution", 270)]
    calls = []

    def fake_ollama(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            [
                {
                    "video_id": 100,
                    "classification": "full_track",
                    "matched_track_position": "A1",
                    "confidence": 0.95,
                    "reasoning": "Exact title and duration match",
                }
            ]
        )

    results = ym.resolve_release(
        "Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama, force_llm=True
    )

    assert len(calls) == 1
    assert results[0].track_id == 1
    assert results[0].llm_confidence == 0.95


def test_resolve_release_no_tracks_skips_llm():
    # Nothing to fuzzy-match against, so there's no point asking the LLM either.
    videos = [_video(100, "Some Video", 9999)]

    results = ym.resolve_release("Some Release", ["Artist"], [], videos, call_ollama=_unreachable_ollama)

    assert results[0].video_type == "unrelated"
    assert results[0].track_id is None


def test_resolve_release_duration_prefilter_skips_llm_when_all_tracks_implausible():
    # Even an exact title match is disqualified by a >20% duration deviance,
    # and with no other track to fall back on, this must resolve to
    # "unrelated" without ever consulting the LLM (nothing plausible to ask).
    tracks = [_track(1, "A1", "Rave-O-Lution", 200)]
    videos = [_video(100, "Rave-O-Lution", 9999)]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert results[0].video_type == "unrelated"
    assert results[0].track_id is None
    assert results[0].fuzzy_score == 0.0


def test_resolve_release_duration_prefilter_lets_plausible_track_win_over_implausible_one():
    # Track A1's duration rules it out outright despite a decent title match;
    # A2 has a plausible duration and a strong (if not exact) title match, so
    # it should win cleanly instead of the video being escalated to the LLM.
    tracks = [
        _track(1, "A1", "Rave-O-Lution", 9999),
        _track(2, "A2", "Rave-O-Lution", 200),
    ]
    videos = [_video(100, "Rave-O-Lution", 200)]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert results[0].video_type == "full_track"
    assert results[0].track_id == 2


def test_resolve_release_low_fuzzy_score_still_escalates_to_llm():
    # A low fuzzy score alone is no longer a reliable enough signal to reject
    # a video without ever asking the LLM (fuzzy score can be misleading on a
    # noisy raw title) - it must always get an LLM look before being rejected.
    # Duration kept within DURATION_DEVIANCE_THRESHOLD of the track's so this
    # is testing the title-based ambiguous case, not the duration pre-filter
    # (which has its own dedicated tests below).
    tracks = [_track(1, "A1", "Rave-O-Lution", 270)]
    videos = [_video(100, "Totally Unrelated Video About Cats", 280)]

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "unrelated", "matched_track_position": None, "confidence": 0.9, "reasoning": "no match"}
        )

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama)

    assert results[0].video_type == "unrelated"
    assert results[0].track_id is None
    assert results[0].llm_confidence == 0.9


def test_resolve_release_chapters_no_longer_force_compilation_classification():
    # Chapter count used to short-circuit straight to "compilation_or_mix" before
    # any fuzzy scoring happened. That heuristic was removed: chapters are just
    # inert metadata now, so a video with many chapters but a strong title match
    # is still evaluated (and accepted) purely on fuzzy score.
    tracks = [_track(1, "A1", "Rave-O-Lution", 270)]
    videos = [_video(100, "Rave-O-Lution", 270, chapters=[{"title": "1"}, {"title": "2"}, {"title": "3"}])]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert results[0].video_type == "full_track"
    assert results[0].track_id == 1


def test_resolve_release_chapters_do_not_prevent_llm_escalation():
    tracks = [_track(1, "A1", "Track One", 270)]
    videos = [_video(100, "Completely Unrelated Video", 280, chapters=[{"title": "1"}, {"title": "2"}, {"title": "3"}])]

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "unrelated", "matched_track_position": None, "confidence": 0.9, "reasoning": "no match"}
        )

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama)

    # Chapter count doesn't short-circuit anything; the low fuzzy score just
    # means it's the LLM's call, not chapters.
    assert results[0].video_type == "unrelated"
    assert results[0].track_id is None
    assert results[0].llm_confidence == 0.9


def test_resolve_release_confident_threshold_is_configurable():
    # A score of ~0.75 is below the default 0.8 threshold (would be ambiguous),
    # but above a custom, lower threshold passed in explicitly. No duration
    # given (deliberately) so the duration-deviance pre-filter can't zero it.
    tracks = [_track(1, "A1", "Rave-O-Lution")]
    videos = [_video(100, "Rave Nation")]

    score = ym.fuzzy_score("Rave-O-Lution", None, None, "Rave Nation", None)
    assert 0.5 < score < ym.CONFIDENT_THRESHOLD

    results = ym.resolve_release(
        "Some Release",
        ["Artist"],
        tracks,
        videos,
        call_ollama=_unreachable_ollama,
        confident_threshold=score - 0.01,
    )

    assert results[0].video_type == "full_track"
    assert results[0].track_id == 1
    assert results[0].llm_confidence is None


def test_resolve_release_multiple_confident_candidates_for_same_track():
    tracks = [_track(1, "A1", "Rave-O-Lution", 270)]
    videos = [
        _video(100, "Rave-O-Lution", 270),
        _video(101, "Rave-O-Lution", 268),
    ]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert len(results) == 2
    assert {r.track_id for r in results} == {1}
    selected = [r for r in results if r.is_selected]
    assert len(selected) == 1


def test_resolve_release_near_tie_does_not_auto_match_wrong_track():
    # Real-world case (release 129318, "Product" by Tenth Chapter): a video
    # titled after one mix scores identically (0.855) against both the wrong
    # track (A, "Original Mix") and another wrong track (AA2, "Bedrock Dub
    # Mix"), while the actually-correct track (AA1, "Daydreamers Tonight At
    # Noon Mix") scores slightly lower (0.78) purely due to quoting/spelling
    # differences from the raw YouTube title. Without a tie check, DB
    # iteration order silently picks the wrong tied track as "confident" and
    # never even asks the LLM. Verified via fuzzy_score() directly.
    #
    # With "Original Mix" also stripped from track titles (denoise_terms
    # applies symmetrically, not just to video titles) and the denoise-retry
    # on the video side, this now resolves correctly via fuzzy alone (no LLM
    # needed) - fake_ollama would still catch a wrong classification if a
    # future change made it ambiguous again.
    tracks = [
        _track(1, "A", "Product (Original Mix)"),
        _track(2, "AA1", 'Product ("Daydreamers Tonight At Noon" Mix)'),
        _track(3, "AA2", 'Product ("Bedrock Dub" Mix)'),
    ]
    videos = [_video(100, "Tenth Chapter -  Product (Daydreamers Tonite At Noon Mix)", 508)]

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "full_track", "matched_track_position": "AA1", "confidence": 0.95, "reasoning": "ok"}
        )

    results = ym.resolve_release("Product", ["Tenth Chapter"], tracks, videos, call_ollama=fake_ollama)

    assert results[0].track_id == 2


def test_resolve_release_clear_winner_above_tie_margin_still_skips_llm():
    # Sanity check that the tie-margin check doesn't just make everything
    # ambiguous: a video with a clearly-dominant best track (no close
    # runner-up) still takes the confident shortcut as before.
    tracks = [_track(1, "A1", "Rave-O-Lution", 270), _track(2, "A2", "Something Else", 200)]
    videos = [_video(100, "Rave-O-Lution", 270)]

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=_unreachable_ollama)

    assert results[0].track_id == 1
    assert results[0].video_type == "full_track"
    assert results[0].llm_confidence is None


# --- resolve_release: ambiguous band escalates to LLM -----------------------


def test_resolve_release_ambiguous_score_calls_llm_and_uses_result():
    tracks = [
        _track(1, "A1", "Hello San Francisco", 439),
        _track(2, "A2", "Rave-O-Lution", 270),
    ]
    # Deliberately mediocre title match -> lands in the ambiguous band.
    videos = [_video(100, "Dance 2 Trance Live Set Excerpt", 440)]

    def fake_ollama(prompt: str) -> str:
        assert "Hello San Francisco" in prompt
        return json.dumps(
            [
                {
                    "video_id": 100,
                    "classification": "full_track",
                    "matched_track_position": "A1",
                    "confidence": 0.9,
                    "reasoning": "Matches duration and artist context",
                }
            ]
        )

    results = ym.resolve_release("Some Release", ["Dance 2 Trance"], tracks, videos, call_ollama=fake_ollama)

    assert len(results) == 1
    assert results[0].track_id == 1
    assert results[0].llm_confidence == 0.9
    assert results[0].video_type == "full_track"


def test_resolve_release_denoise_retry_flips_ambiguous_winner_and_uses_denoised_title_for_llm(monkeypatch):
    # Real-world bug this guards against: a video title carrying repeated
    # uploader boilerplate (artist/release/year, identical across every video
    # from that channel) can make an unrelated track win the raw fuzzy
    # comparison purely because of that noise. After stripping the boilerplate,
    # the correct track should win instead — and if it's still ambiguous even
    # then, the LLM must see the denoised title, not the raw noisy one.
    tracks = [
        _track(1, "A1", "Correct Track", 200),
        _track(2, "A2", "Wrong Track", 200),
    ]
    videos = [_video(100, "NOISY ARTIST NOISY RELEASE Correct Track NOISY ARTIST", 200)]

    def fake_fuzzy_score(track_title, track_artist, track_duration, video_title, video_duration):
        noisy = "NOISY ARTIST" in video_title or "NOISY RELEASE" in video_title
        if noisy:
            # Raw, boilerplate-laden title: the wrong track "wins", ambiguous band.
            return {"Correct Track": 0.5, "Wrong Track": 0.6}[track_title]
        # Denoised title: the correct track now wins, but still ambiguous -> LLM.
        return {"Correct Track": 0.65, "Wrong Track": 0.4}[track_title]

    monkeypatch.setattr(ym, "fuzzy_score", fake_fuzzy_score)

    prompts = []

    def fake_ollama(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {"classification": "full_track", "matched_track_position": "A1", "confidence": 0.9, "reasoning": "ok"}
        )

    results = ym.resolve_release(
        "Noisy Release", ["Noisy Artist"], tracks, videos, call_ollama=fake_ollama
    )

    assert len(prompts) == 1
    assert "NOISY ARTIST" not in prompts[0]
    assert "NOISY RELEASE" not in prompts[0]
    assert "Correct Track" in prompts[0]
    assert results[0].track_id == 1


def test_resolve_release_denoise_retry_confident_result_skips_llm(monkeypatch):
    tracks = [_track(1, "A1", "Correct Track", 200)]
    videos = [_video(100, "NOISY ARTIST NOISY RELEASE Correct Track", 200)]

    def fake_fuzzy_score(track_title, track_artist, track_duration, video_title, video_duration):
        noisy = "NOISY ARTIST" in video_title or "NOISY RELEASE" in video_title
        return 0.5 if noisy else 0.9  # ambiguous raw, confident denoised

    monkeypatch.setattr(ym, "fuzzy_score", fake_fuzzy_score)

    results = ym.resolve_release(
        "Noisy Release", ["Noisy Artist"], tracks, videos, call_ollama=_unreachable_ollama
    )

    assert results[0].track_id == 1
    assert results[0].video_type == "full_track"
    assert results[0].llm_confidence is None


def test_resolve_release_degenerate_denoised_title_falls_back_to_raw_title_for_llm():
    # Real-world case (release 129318): a video titled just "Tenth Chapter -
    # Product" (artist + release title, nothing else) denoises down to "-" -
    # using that for the LLM prompt throws away the one signal the LLM could
    # otherwise reason from (confirmed empirically: the degenerate title made
    # the LLM's answer flip between correct and "unrelated" across identical
    # calls). It must fall back to the original, informative raw title.
    tracks = [_track(1, "A1", "Alpha Beta", 200), _track(2, "A2", "Gamma Delta", 200)]
    videos = [_video(100, "Plugh Corp - Xyzzy Record", 200)]

    prompts = []

    def fake_ollama(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {"classification": "full_track", "matched_track_position": "A1", "confidence": 0.9, "reasoning": "ok"}
        )

    ym.resolve_release(
        "Xyzzy Record", ["Plugh Corp"], tracks, videos, call_ollama=fake_ollama
    )

    assert len(prompts) == 1
    assert 'title "Plugh Corp - Xyzzy Record"' in prompts[0]


def test_resolve_release_custom_denoise_terms_are_used_in_retry(monkeypatch):
    # A generic, non-release-specific term (configured via denoise_terms,
    # not derived from the release/artist) should be stripped the same way
    # as release/artist boilerplate is.
    tracks = [_track(1, "A1", "Correct Track", 200)]
    videos = [_video(100, "Correct Track (Custom Boilerplate Term)", 200)]

    def fake_fuzzy_score(track_title, track_artist, track_duration, video_title, video_duration):
        noisy = "Custom Boilerplate Term" in video_title
        return 0.5 if noisy else 0.9  # ambiguous raw, confident after stripping the extra term

    monkeypatch.setattr(ym, "fuzzy_score", fake_fuzzy_score)

    results = ym.resolve_release(
        "Some Release",
        ["Some Artist"],
        tracks,
        videos,
        call_ollama=_unreachable_ollama,
        denoise_terms=["Custom Boilerplate Term"],
    )

    assert results[0].track_id == 1
    assert results[0].video_type == "full_track"
    assert results[0].llm_confidence is None


def test_resolve_release_multiple_ambiguous_videos_get_separate_llm_calls():
    # Regression test: a small local model can silently ignore all but one item
    # in a batched multi-video prompt. Ambiguous videos must never be batched
    # into a single call — each gets its own prompt, and the fake here only
    # ever answers for exactly the one video_id it was actually asked about,
    # mimicking a model that can't reliably handle more than one item at once.
    tracks = [
        _track(1, "A1", "Hello San Francisco", 439),
        _track(2, "A2", "Rave-O-Lution", 270),
        _track(3, "A3", "Something Else Entirely", 200),
    ]
    videos = [
        _video(100, "Dance 2 Trance Live Set Excerpt", 440),
        _video(101, "Live Radio Excerpt Mix", 275),
        _video(102, "Some Random Clip Here", 205),
    ]
    prompts_seen = []

    def fake_ollama(prompt: str) -> str:
        prompts_seen.append(prompt)
        if "Dance 2 Trance Live Set Excerpt" in prompt:
            position = "A1"
        elif "Live Radio Excerpt Mix" in prompt:
            position = "A2"
        else:
            position = "A3"
        return json.dumps(
            {
                "classification": "full_track",
                "matched_track_position": position,
                "confidence": 0.8,
                "reasoning": "context match",
            }
        )

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama)

    # One call per ambiguous video, not one call for the whole batch.
    assert len(prompts_seen) == 3
    by_video = {r.video_id: r for r in results}
    assert by_video[100].track_id == 1
    assert by_video[101].track_id == 2
    assert by_video[102].track_id == 3
    assert all(r.llm_confidence == 0.8 for r in results)


def test_resolve_release_llm_pick_overridden_by_much_stronger_fuzzy_alternative(monkeypatch):
    # Real-world case (release 13667935, Arabic-script tracklist): the video's
    # own title/description clearly named track B1, and fuzzy scoring agreed
    # (0.62 for B1 vs 0.44 for A1) - but the LLM confidently (deterministically,
    # not a fluke) claimed A1 instead, with a vague, factually wrong
    # justification. Since nothing else competes for A1, _select_best_per_track
    # has no alternative to blend against, so the guard has to live in the LLM
    # classification step itself: don't commit to a claim that's clearly
    # contradicted by a stronger fuzzy alternative on the very same video -
    # fall back to that alternative (B1) rather than discarding to
    # "unrelated", since it's already the best-scoring candidate overall.
    tracks = [_track(1, "A1", "Track A1 Title", 200), _track(2, "B1", "Track B1 Title", 200)]
    videos = [_video(100, "Ambiguous Video Title", 200)]

    def fake_fuzzy_score(track_title, track_artist, track_duration, video_title, video_duration):
        return {"Track A1 Title": 0.44, "Track B1 Title": 0.62}[track_title]

    monkeypatch.setattr(ym, "fuzzy_score", fake_fuzzy_score)

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "full_track", "matched_track_position": "A1", "confidence": 0.9, "reasoning": "vague"}
        )

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama)

    assert results[0].track_id == 2
    assert results[0].video_type == "full_track"
    assert results[0].fuzzy_score == 0.62
    assert results[0].llm_confidence is None
    assert "falling back" in results[0].video_type_reasoning


def test_resolve_release_llm_pick_trusted_when_no_stronger_fuzzy_alternative(monkeypatch):
    # Sanity check the guard doesn't fire when there's nothing to distrust it
    # against: the LLM's claimed track IS the video's own fuzzy-best guess.
    tracks = [_track(1, "A1", "Track A1 Title", 200), _track(2, "B1", "Track B1 Title", 200)]
    videos = [_video(100, "Ambiguous Video Title", 200)]

    def fake_fuzzy_score(track_title, track_artist, track_duration, video_title, video_duration):
        return {"Track A1 Title": 0.6, "Track B1 Title": 0.4}[track_title]

    monkeypatch.setattr(ym, "fuzzy_score", fake_fuzzy_score)

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "full_track", "matched_track_position": "A1", "confidence": 0.9, "reasoning": "ok"}
        )

    results = ym.resolve_release("Some Release", ["Artist"], tracks, videos, call_ollama=fake_ollama)

    assert results[0].track_id == 1
    assert results[0].video_type == "full_track"


def test_resolve_release_llm_bare_object_response_is_handled():
    # Some models return a single bare object instead of a one-element array
    # when there's only one ambiguous video to classify in the batch.
    tracks = [
        _track(1, "A1", "Hello San Francisco", 439),
        _track(2, "A2", "Rave-O-Lution", 270),
    ]
    videos = [_video(100, "Dance 2 Trance Live Set Excerpt", 440)]

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {
                "video_id": 100,
                "classification": "full_track",
                "matched_track_position": "A1",
                "confidence": 0.9,
                "reasoning": "Matches duration and artist context",
            }
        )

    results = ym.resolve_release("Some Release", ["Dance 2 Trance"], tracks, videos, call_ollama=fake_ollama)

    assert len(results) == 1
    assert results[0].track_id == 1
    assert results[0].llm_confidence == 0.9


def test_resolve_release_llm_failure_falls_back_to_fuzzy_best_effort():
    tracks = [_track(1, "A1", "Hello San Francisco", 439)]
    videos = [_video(100, "Dance 2 Trance Live Set Excerpt", 440)]

    def failing_ollama(prompt: str) -> str:
        raise RuntimeError("ollama not running")

    results = ym.resolve_release("Some Release", ["Dance 2 Trance"], tracks, videos, call_ollama=failing_ollama)

    assert len(results) == 1
    assert results[0].llm_confidence is None
    # Without an LLM verdict there's no reliable basis to commit an already-
    # ambiguous fuzzy guess to a track; it's left unassigned, not silently
    # promoted to full_track just because a best_track happened to exist.
    assert results[0].video_type == "unrelated"
    assert results[0].track_id is None


def test_resolve_release_llm_malformed_json_falls_back():
    tracks = [_track(1, "A1", "Hello San Francisco", 439)]
    videos = [_video(100, "Dance 2 Trance Live Set Excerpt", 440)]

    results = ym.resolve_release(
        "Some Release", ["Dance 2 Trance"], tracks, videos, call_ollama=lambda p: "not json"
    )

    assert len(results) == 1
    assert results[0].llm_confidence is None


# --- match_videos: end-to-end against a seeded DB ---------------------------


def seed_release(conn, release_id, tracks, videos):
    data = ReleaseData(
        id=release_id,
        title="Test Release",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=release_id * 10, name="Test Artist")],
        labels=[],
        tracks=tracks,
        videos=videos,
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")


def test_match_videos_end_to_end_clean_single_track(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "Someone", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)

    video_row = conn.execute("SELECT * FROM release_videos WHERE release_id = 1").fetchone()
    assert video_row["video_type"] == "full_track"
    assert video_row["yt_title"] == "Rave-O-Lution"

    match_row = conn.execute("SELECT * FROM track_video_matches").fetchone()
    assert match_row["is_selected"] == 1
    assert match_row["fuzzy_score"] > 0.9


def test_match_videos_backfills_missing_track_duration_from_matched_video(conn):
    # Real gap: Discogs tracklist data sometimes lacks duration entirely.
    # Once a video is confidently matched, its duration is a reasonable
    # stand-in worth writing back onto the track.
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration=None, artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)

    track_row = conn.execute("SELECT duration FROM tracks WHERE release_id = 1").fetchone()
    assert track_row["duration"] == "4:30"


def test_match_videos_does_not_overwrite_existing_track_duration(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="9:99", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)

    track_row = conn.execute("SELECT duration FROM tracks WHERE release_id = 1").fetchone()
    assert track_row["duration"] == "9:99"


def test_clear_matches_resets_video_type_and_deletes_match_rows(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "Someone", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)
    assert conn.execute("SELECT COUNT(*) AS n FROM track_video_matches").fetchone()["n"] == 1

    ym.clear_matches(conn, 1)

    assert conn.execute("SELECT COUNT(*) AS n FROM track_video_matches").fetchone()["n"] == 0
    video_row = conn.execute("SELECT * FROM release_videos WHERE release_id = 1").fetchone()
    assert video_row["video_type"] is None
    assert video_row["video_type_reasoning"] is None
    assert video_row["best_fuzzy_score"] is None
    assert video_row["best_fuzzy_track_position"] is None
    # Fetched YouTube metadata itself is untouched.
    assert video_row["yt_title"] == "Rave-O-Lution"
    assert video_row["yt_duration"] == 270


def test_clear_matches_only_affects_the_given_release(conn):
    seed_release(
        conn, 1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )
    seed_release(
        conn, 2,
        tracks=[TrackData(position="A1", title="Hello San Francisco", duration="7:16", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=xyz", duration=439, description=None)],
    )

    def fake_extractor(url):
        if "abc" in url:
            return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}
        return {"title": "Hello San Francisco", "duration": 439, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)

    ym.clear_matches(conn, 1)

    assert conn.execute("SELECT video_type FROM release_videos WHERE release_id = 1").fetchone()["video_type"] is None
    assert conn.execute("SELECT video_type FROM release_videos WHERE release_id = 2").fetchone()["video_type"] == "full_track"


def test_match_videos_ignores_chapter_count_and_escalates_low_score_to_llm(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=90, description=None)],
    )

    def fake_extractor(url):
        return {
            "title": "xyzxyzxyz",
            "duration": 90,
            "uploader": "Label",
            "description": "",
            "tags": [],
            "chapters": [{"title": "1"}, {"title": "2"}, {"title": "3"}],
        }

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            {"classification": "unrelated", "matched_track_position": None, "confidence": 0.9, "reasoning": "no match"}
        )

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=fake_ollama)

    video_row = conn.execute("SELECT * FROM release_videos WHERE release_id = 1").fetchone()
    assert video_row["video_type"] == "unrelated"
    assert conn.execute("SELECT COUNT(*) AS n FROM track_video_matches").fetchone()["n"] == 0


def test_count_pending_releases_counts_releases_with_unclassified_videos(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )
    seed_release(
        conn,
        2,
        tracks=[TrackData(position="A1", title="Track Two", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=def", duration=180, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Track", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    assert ym.count_pending_releases(conn) == 2
    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)
    assert ym.count_pending_releases(conn) == 0


def test_match_videos_skips_already_matched_releases_by_default(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )
    calls = []

    def fake_extractor(url):
        calls.append(url)
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)
    assert len(calls) == 1

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)
    assert len(calls) == 1  # not re-fetched


def test_match_videos_force_reprocesses(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )
    calls = []

    def fake_extractor(url):
        calls.append(url)
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)
    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama, force=True)
    assert len(calls) == 2


def test_match_videos_reprocess_without_force_recomputes_but_skips_refetch(conn):
    # "Clean up" mode: reprocess classification for already-classified releases,
    # but never re-fetch metadata for videos that already have it (fast, no
    # network calls beyond whatever Ollama needs for genuinely ambiguous cases).
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )
    fetch_calls = []
    resolve_calls = []

    def fake_extractor(url):
        fetch_calls.append(url)
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    def counting_ollama(prompt):
        resolve_calls.append(prompt)
        raise AssertionError("should not be ambiguous in this test")

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=counting_ollama)
    assert len(fetch_calls) == 1

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=counting_ollama, force=False, reprocess=True)
    assert len(fetch_calls) == 1  # not re-fetched
    row = conn.execute("SELECT video_type FROM release_videos WHERE release_id = 1").fetchone()
    assert row["video_type"] == "full_track"  # still reprocessed successfully


def test_match_videos_isolates_single_video_fetch_failure(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[
            VideoData(title="ignored", url="https://youtube.com/watch?v=good", duration=180, description=None),
            VideoData(title="ignored", url="https://youtube.com/watch?v=bad", duration=180, description=None),
        ],
    )

    def fake_extractor(url):
        if "bad" in url:
            raise RuntimeError("410 Gone")
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    errors = []
    ym.match_videos(
        conn,
        extractor=fake_extractor,
        call_ollama=_unreachable_ollama,
        on_error=lambda release_id, exc: errors.append((release_id, str(exc))),
    )

    assert len(errors) == 1
    assert "bad" in errors[0][1]

    rows = conn.execute("SELECT url, video_type FROM release_videos WHERE release_id = 1").fetchall()
    by_url = {row["url"]: row["video_type"] for row in rows}
    assert by_url["https://youtube.com/watch?v=good"] == "full_track"
    assert by_url["https://youtube.com/watch?v=bad"] == "unrelated"


def test_match_videos_treats_none_extractor_result_as_a_fetch_failure(conn):
    """yt-dlp's extract_info() can return None without raising (observed on
    a real release) - must be treated the same as a genuine exception, not
    silently written as an all-NULL "successfully fetched, nothing here"
    row that then gets permanently excluded from matching - see
    fetch_video_metadata's docstring."""
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[
            VideoData(title="ignored", url="https://youtube.com/watch?v=good", duration=180, description=None),
            VideoData(title="ignored", url="https://youtube.com/watch?v=none", duration=180, description=None),
        ],
    )

    def fake_extractor(url):
        if "none" in url:
            return None
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    errors = []
    ym.match_videos(
        conn,
        extractor=fake_extractor,
        call_ollama=_unreachable_ollama,
        on_error=lambda release_id, exc: errors.append((release_id, str(exc))),
    )

    assert len(errors) == 1
    assert "none" in errors[0][1]

    rows = conn.execute("SELECT url, yt_title, video_type FROM release_videos WHERE release_id = 1").fetchall()
    by_url = {row["url"]: (row["yt_title"], row["video_type"]) for row in rows}
    assert by_url["https://youtube.com/watch?v=good"][1] == "full_track"
    assert by_url["https://youtube.com/watch?v=none"] == (None, "unrelated")


def test_match_videos_stops_early_when_should_stop_set(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )
    seed_release(
        conn,
        2,
        tracks=[TrackData(position="A1", title="Track Two", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=def", duration=180, description=None)],
    )
    calls = []

    def fake_extractor(url):
        calls.append(url)
        return {"title": "Track", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama, should_stop=lambda: True)
    assert calls == []


def test_match_videos_reports_progress(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=180, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Track One", "duration": 180, "uploader": "X", "description": "", "tags": [], "chapters": []}

    calls = []
    ym.match_videos(
        conn,
        extractor=fake_extractor,
        call_ollama=_unreachable_ollama,
        on_progress=lambda *args: calls.append(args),
    )

    assert calls == [(1, 1, 1)]


def test_debug_score_matrix_flags_fuzzy_only_selection(conn):
    seed_release(
        conn,
        1,
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=_unreachable_ollama)

    matrix = ym.debug_score_matrix(conn, 1)

    assert len(matrix) == 1
    cell = matrix[0]["cells"][0]
    assert cell["is_selected"] is True
    assert cell["is_llm_selected"] is False


def test_debug_score_matrix_flags_llm_assisted_selection(conn):
    seed_release(
        conn,
        1,
        tracks=[
            TrackData(position="A1", title="Hello San Francisco", duration="7:19", artist=None),
        ],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=440, description=None)],
    )

    def fake_extractor(url):
        return {
            "title": "Dance 2 Trance Live Set Excerpt",
            "duration": 440,
            "uploader": "X",
            "description": "",
            "tags": [],
            "chapters": [],
        }

    def fake_ollama(prompt: str) -> str:
        return json.dumps(
            [
                {
                    "video_id": 1,
                    "classification": "full_track",
                    "matched_track_position": "A1",
                    "confidence": 0.9,
                    "reasoning": "matches on context",
                }
            ]
        )

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=fake_ollama)

    matrix = ym.debug_score_matrix(conn, 1)

    assert len(matrix) == 1
    cell = matrix[0]["cells"][0]
    assert cell["is_selected"] is True
    assert cell["is_llm_selected"] is True


# --- _select_best_per_track: blended fuzzy+LLM selection key ----------------


def _match(video_id, track_id, fuzzy_score, llm_confidence=None):
    return ym.MatchResult(
        video_id=video_id,
        track_id=track_id,
        fuzzy_score=fuzzy_score,
        video_type="full_track",
        llm_confidence=llm_confidence,
    )


def test_select_best_per_track_prefers_higher_fuzzy_score_over_marginally_higher_llm_confidence():
    # Real-world case (release 64733, track A4 "Zarah..."): two LLM-confirmed
    # candidates, where raw llm_confidence (1.0 vs 0.99 - noise from a small
    # model) previously overrode a real fuzzy_score gap (0.634 vs 0.665) that
    # clearly favors the second video.
    weaker_fuzzy = _match(2838, track_id=1, fuzzy_score=0.63385714285714279, llm_confidence=1.0)
    stronger_fuzzy = _match(2842, track_id=1, fuzzy_score=0.6654409201812479, llm_confidence=0.99)

    ym._select_best_per_track([weaker_fuzzy, stronger_fuzzy])

    assert stronger_fuzzy.is_selected is True
    assert weaker_fuzzy.is_selected is False


def test_select_best_per_track_prefers_clean_fuzzy_match_over_misclassified_llm_match():
    # Real-world case (release 129318, track AA1): a clean, unambiguous
    # fuzzy-only match (no LLM needed) previously lost to a video the LLM
    # had misclassified onto this track, despite that video's own fuzzy
    # score against this specific track being weak (0.349 - it actually
    # belongs to a different track).
    clean_fuzzy_match = _match(619, track_id=2, fuzzy_score=0.855)
    misclassified_llm_match = _match(618, track_id=2, fuzzy_score=0.3488, llm_confidence=0.9)

    ym._select_best_per_track([clean_fuzzy_match, misclassified_llm_match])

    assert clean_fuzzy_match.is_selected is True
    assert misclassified_llm_match.is_selected is False


def test_select_best_per_track_fuzzy_only_candidates_unaffected():
    # No LLM involvement at all: behaves exactly as before (plain fuzzy_score
    # comparison).
    lower = _match(100, track_id=1, fuzzy_score=0.8)
    higher = _match(101, track_id=1, fuzzy_score=0.9)

    ym._select_best_per_track([lower, higher])

    assert higher.is_selected is True
    assert lower.is_selected is False


# --- build_extractor: optional cookies-from-browser auth ---------------------


class _FakeYoutubeDL:
    captured_opts = None

    def __init__(self, opts):
        _FakeYoutubeDL.captured_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        return {"title": "ok"}


def test_build_extractor_without_cookies_browser_omits_cookiesfrombrowser(monkeypatch):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)

    extractor = ym.build_extractor()
    extractor("https://youtube.com/watch?v=abc")

    assert "cookiesfrombrowser" not in _FakeYoutubeDL.captured_opts


def test_build_extractor_with_cookies_browser_sets_cookiesfrombrowser(monkeypatch):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)

    extractor = ym.build_extractor("firefox")
    extractor("https://youtube.com/watch?v=abc")

    assert _FakeYoutubeDL.captured_opts["cookiesfrombrowser"] == ("firefox",)
