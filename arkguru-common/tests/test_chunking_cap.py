"""Sentence split + hard token cap (bge-m3 512)."""
from __future__ import annotations

from common.chunking import pack_windows, split_for_packing, split_sentences
from common.tokenizer import DEFAULT_MAX_TOKENS, count_tokens


def test_split_sentences_danda_and_newline():
    text = "प्रथम भाव। द्वितीय भाव॥\nThird house follows."
    sents = split_sentences(text)
    joined = " ".join(sents)
    assert "प्रथम भाव" in joined
    assert "द्वितीय भाव" in joined
    assert any("Third house" in s for s in sents)
    assert len(sents) >= 3


def test_split_sentences_no_longer_requires_capital_after_period():
    sents = split_sentences("End of verse. next line starts lower.")
    assert len(sents) >= 2


def test_pack_windows_hard_cap_on_giant_headingless_section():
    # ~190k chars, no headings, no .!?+capital pattern (the live failure mode).
    blob = ("saturnintheseventhhouseandthesubsublord " * 80 + "\n") * 80
    assert len(blob) > 190_000
    units = split_for_packing(blob)
    windows = pack_windows(
        units, target_tokens=400, overlap_tokens=40, min_tokens=80,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    assert windows
    assert all(count_tokens(text) <= DEFAULT_MAX_TOKENS for text, _ov in windows)
    overlapped = [ov for _t, ov in windows[1:]]
    assert overlapped and all(ov > 0 for ov in overlapped)


def test_pack_windows_splits_single_oversized_unit():
    huge = "token " * 2000
    windows = pack_windows(
        [huge], target_tokens=400, overlap_tokens=20, max_tokens=64,
    )
    assert len(windows) > 1
    assert all(count_tokens(text) <= 64 for text, _ov in windows)


def test_abbreviation_merge_still_holds():
    sents = split_sentences("See Fig. 2 for details. Then continue.")
    assert any("Fig. 2" in s for s in sents)
