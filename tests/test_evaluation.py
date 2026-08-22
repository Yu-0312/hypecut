"""Answer keys, matching and the metrics built on them."""

from __future__ import annotations

import pytest

from hypecut.evaluation import Highlight, Labels, load_labels, score_plan, write_labels


def _labels(*spans: tuple[float, float]) -> Labels:
    return Labels(video="v.mp4", highlights=[Highlight(a, b) for a, b in spans])


# --------------------------------------------------------------- the metric


def test_a_clip_hits_when_it_contains_the_moment():
    """Containment, not overlap: finding is a separate question from framing."""
    score = score_plan(_labels((10.0, 20.0)), [(12.0, 30.0)])
    assert score.found == 1
    assert score.recall == 1.0


def test_a_clip_that_only_clips_the_edge_is_a_miss():
    """Overlapping the tail of a moment is not finding it."""
    score = score_plan(_labels((10.0, 20.0)), [(19.0, 30.0)])
    assert score.found == 0
    assert score.missed[0].start == 10.0
    assert score.spurious == [(19.0, 30.0)]


def test_precision_counts_clips_that_matched_nothing():
    score = score_plan(_labels((10.0, 20.0)), [(12.0, 18.0), (60.0, 70.0), (80.0, 90.0)])
    assert score.recall == 1.0
    assert score.precision == pytest.approx(1 / 3)
    assert len(score.spurious) == 2


def test_two_clips_covering_one_moment_are_both_credited():
    """Neither is spurious — they both contain it, so neither is a false alarm."""
    score = score_plan(_labels((10.0, 20.0)), [(5.0, 16.0), (14.0, 25.0)])
    assert score.found == 1
    assert score.spurious == []
    assert score.precision == 1.0


def test_f1_is_zero_when_nothing_was_found():
    score = score_plan(_labels((10.0, 20.0)), [(60.0, 70.0)])
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_an_empty_cut_list_scores_zero_without_dividing_by_zero():
    score = score_plan(_labels((10.0, 20.0)), [])
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------ coverage


def test_coverage_separates_finding_from_framing():
    """Same recall, different framing — the metric has to tell them apart."""
    tight = score_plan(_labels((10.0, 20.0)), [(14.0, 16.0)])
    whole = score_plan(_labels((10.0, 20.0)), [(8.0, 22.0)])

    assert tight.recall == whole.recall == 1.0
    assert tight.coverage == pytest.approx(0.2)
    assert whole.coverage == pytest.approx(1.0)


def test_coverage_ignores_moments_that_were_never_found():
    """Otherwise a miss would drag coverage down and confuse the two failures."""
    score = score_plan(_labels((10.0, 20.0), (100.0, 110.0)), [(8.0, 22.0)])
    assert score.recall == 0.5
    assert score.coverage == pytest.approx(1.0)


# ---------------------------------------------------------------- labels I/O


def test_a_draft_round_trips_and_undecided_entries_are_not_answers(tmp_path):
    dest = tmp_path / "draft.yaml"
    write_labels(
        Labels(video="v.mp4", highlights=[], annotator="max"),
        dest,
        draft=[
            {"start": 1.0, "end": 5.0, "keep": None},
            {"start": 10.0, "end": 20.0, "keep": True},
            {"start": 30.0, "end": 40.0, "keep": False},
        ],
    )

    labels = load_labels(dest)
    assert [(h.start, h.end) for h in labels.highlights] == [(10.0, 20.0)]
    assert labels.annotator == "max"


def test_a_bare_entry_counts_as_kept(tmp_path):
    """Hand-written keys should not need `keep: true` on every line."""
    dest = tmp_path / "hand.yaml"
    dest.write_text("video: v.mp4\nhighlights:\n  - {start: 3, end: 9}\n", encoding="utf-8")
    assert len(load_labels(dest).highlights) == 1


def test_highlights_come_back_in_time_order(tmp_path):
    dest = tmp_path / "shuffled.yaml"
    dest.write_text(
        "video: v.mp4\nhighlights:\n  - {start: 50, end: 60}\n  - {start: 5, end: 9}\n",
        encoding="utf-8",
    )
    assert [h.start for h in load_labels(dest).highlights] == [5.0, 50.0]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("highlights:\n  - {start: 1, end: 2}\n", "no `video`"),
        ("video: v.mp4\nhighlights: []\n", "no highlights marked"),
        ("video: v.mp4\nhighlights:\n  - {start: 9, end: 3}\n", "ends before it starts"),
        ("video: v.mp4\nhighlights:\n  - {start: 1}\n", "numeric start and end"),
    ],
)
def test_unusable_label_files_are_rejected_with_a_reason(tmp_path, body, match):
    dest = tmp_path / "bad.yaml"
    dest.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_labels(dest)


def test_an_all_undecided_draft_is_rejected(tmp_path):
    """A draft nobody has been through is not an answer key."""
    dest = tmp_path / "untouched.yaml"
    write_labels(
        Labels(video="v.mp4", highlights=[]), dest, draft=[{"start": 1.0, "end": 5.0, "keep": None}]
    )
    with pytest.raises(ValueError, match="no highlights marked"):
        load_labels(dest)


def test_json_labels_work_too(tmp_path):
    dest = tmp_path / "labels.json"
    write_labels(Labels(video="v.mp4", highlights=[Highlight(1.0, 4.0, "ace")]), dest)
    labels = load_labels(dest)
    assert labels.highlights[0].label == "ace"
