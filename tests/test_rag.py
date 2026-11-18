"""Document retrieval for the assistant's search_docs tool (FR-12). Offline and deterministic:
BM25 over the repo's own markdown, no model and no API key."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from replenishai import rag

ROOT = Path(__file__).resolve().parent.parent


def test_tokenize_mixes_words_and_chinese_bigrams():
    toks = rag.tokenize("LightGBM 用 Poisson 目標，FOODS_1_001 的資料洩漏")
    assert {"lightgbm", "poisson", "foods_1_001"} <= set(toks)
    assert {"目標", "資料", "料洩", "洩漏"} <= set(toks)
    assert "用" in toks                                    # a lone CJK character is kept


def test_split_markdown_follows_headings_and_ignores_fences():
    md = "# Title\nintro\n## A\nalpha text\n```\n# not a heading\n```\n### A1\ndeep\n## B\nbeta\n"
    chunks = rag.split_markdown("x.md", md)
    assert [c.section for c in chunks] == ["Title", "Title > A", "Title > A > A1", "Title > B"]
    assert "# not a heading" in chunks[1].text
    assert [c.id for c in chunks] == ["x.md#0", "x.md#1", "x.md#2", "x.md#3"]


def test_long_sections_are_split_into_bounded_chunks():
    para = "段落內容。" * 60                                    # ~300 chars
    chunks = rag.split_markdown("x.md", "# T\n" + "\n\n".join([para] * 6))
    assert len(chunks) > 1
    assert all(len(c.text) <= rag.HARD_CHUNK_CHARS for c in chunks)
    assert "".join(c.text.replace("\n", "") for c in chunks) == para * 6   # nothing lost


def test_corpus_covers_the_project_docs():
    sources = {c.source for c in rag.get_index().chunks}
    assert {"README.md", "project-brief.md", "docs/04-sad.md", "docs/deploy-railway.md"} <= sources
    assert all(len(c.text) <= rag.HARD_CHUNK_CHARS for c in rag.get_index().chunks)


@pytest.mark.parametrize("query, source", [
    ("Railway 部署", "docs/deploy-railway.md"),
    ("補貨公式 再訂購點 安全庫存", "docs/05-sds.md"),
])
def test_obvious_queries_hit_the_right_document(query, source):
    hits = rag.search([query])["hits"]
    assert source in {h["source"] for h in hits}


def test_rrf_rewards_passages_found_by_several_queries():
    chunks = [rag.Chunk(f"t#{i}", "t.md", "S", t) for i, t in
              enumerate(["apple banana", "apple cherry", "banana durian", "egg fig"])]
    idx = rag.Index(chunks)
    out = rag.search(["apple", "banana"], k=4, index=idx)
    assert out["hits"][0]["id"] == "t#0"                   # the only chunk both queries find
    assert "t#3" not in {h["id"] for h in out["hits"]}     # shares no term: never returned


def test_no_match_returns_empty_hits_with_note():
    out = rag.search(["zzzqqq"])
    assert out["hits"] == [] and "沒有找到" in out["note"]
    json.dumps(out, ensure_ascii=False)                    # tool output must be JSON-serialisable


@pytest.mark.parametrize("bad, exc", [
    ("not a list", TypeError),
    ([], ValueError),
    (["", "x"], ValueError),
    (["a", "b", "c", "d"], ValueError),
    (["x" * 201], ValueError),
])
def test_bad_queries_are_rejected(bad, exc):
    with pytest.raises(exc):
        rag.search(bad)


def test_docs_eval_cases_are_answerable_from_retrieval():
    """Every FR-12 eval case: searching its own question already returns a passage that contains
    one of the strings the grader requires, so a failure there is the model's, not the index's."""
    cases = json.loads((ROOT / "eval" / "assistant_cases.json").read_text(encoding="utf-8"))["cases"]
    docs = [c for c in cases if c["category"] == "docs"]
    assert docs
    for c in docs:
        text = " ".join(h["section"] + h["text"] for h in rag.search([c["question"]])["hits"])
        assert any(s in text for s in c["answer_any"]), c["id"]
