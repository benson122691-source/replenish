"""Project-document retrieval (RAG) for the assistant's `search_docs` tool (FR-12).

Adapted from langchain-ai/rag-research-agent-template (MIT). The template's LangGraph nodes map
onto this project like this:

  template                                          here
  ------------------------------------------------  ---------------------------------------------
  index_graph.index_docs (docs -> vector store)     build_index(): split the project's markdown by
                                                    heading, keep an in-memory BM25 index
  researcher_graph.generate_queries                 the model writes 1-3 queries as the tool's
                                                    `queries` argument
  researcher_graph.retrieve_in_parallel (Send)      search(): rank each query separately, merge
                                                    with reciprocal rank fusion (RRF)
  retrieval_graph router / research plan / respond  SYSTEM_PROMPT in assistant.py; the existing
                                                    tool loop runs the plan (one search per step)

Why BM25 and not embeddings: the corpus is ~50 KB of markdown, BM25 needs no extra service,
API key or dependency, and it is deterministic, so retrieval can be tested offline. The tokenizer
mixes English words with Chinese character bigrams, because the docs are written in both.
Swapping in an embedding retriever only means replacing `Index.rank`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_GLOBS = ("README.md", "project-brief.md", "docs/*.md")

MAX_CHUNK_CHARS = 900        # a section longer than this is split at blank lines
HARD_CHUNK_CHARS = 1500      # a single paragraph longer than this is cut
MAX_QUERIES = 3              # template: generate_queries returns a handful of diverse queries
MAX_QUERY_CHARS = 200
TOP_K = 4                    # passages returned per tool call (keeps the model's context small)
PER_QUERY = 8                # candidates taken from each query before fusion
RRF_K = 60                   # standard reciprocal-rank-fusion constant
BM25_K1, BM25_B = 1.5, 0.75

_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
_WORD = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?")
_CJK = re.compile(r"[㐀-䶿一-鿿]+")


@dataclass(frozen=True)
class Chunk:
    id: str           # "docs/04-sad.md#3"
    source: str       # path relative to the repo root
    section: str      # heading path, "04 SAD：系統架構 > 3. 選型理由"
    text: str


def tokenize(text: str) -> list[str]:
    """English/code words (lower-cased, FOODS_1_001 stays one token) + Chinese character bigrams.
    A lone Chinese character is kept as a unigram so one-character queries still match."""
    low = text.lower()
    toks = _WORD.findall(low)
    for run in _CJK.findall(low):
        toks += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


# ---------------------------------------------------------------- index (template: index_graph)

def _paragraphs(body: str) -> list[str]:
    out, cur = [], []
    for line in body.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            out.append("\n".join(cur)); cur = []
    if cur:
        out.append("\n".join(cur))
    return [p for part in out for p in
            (part[i:i + HARD_CHUNK_CHARS] for i in range(0, len(part), HARD_CHUNK_CHARS))]


def _pack(body: str) -> list[str]:
    """Greedy-pack paragraphs into pieces of at most MAX_CHUNK_CHARS (a long paragraph stays whole)."""
    pieces, cur = [], ""
    for p in _paragraphs(body):
        if cur and len(cur) + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        pieces.append(cur)
    return pieces


def split_markdown(source: str, md: str) -> list[Chunk]:
    """One chunk per heading section (headings inside ``` fences are ignored)."""
    sections: list[tuple[str, list[str]]] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush():
        if any(l.strip() for l in lines):
            sections.append((" > ".join(t for _, t in path), list(lines)))
        lines.clear()

    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path = [(lv, t) for lv, t in path if lv < level] + [(level, m.group(2))]
        else:
            lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for section, body in sections:
        for piece in _pack("\n".join(body)):
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, section or source, piece))
    return chunks


def load_corpus(root: Path = ROOT) -> list[Chunk]:
    files = sorted({p for g in DOC_GLOBS for p in root.glob(g) if p.is_file()})
    return [c for f in files
            for c in split_markdown(f.relative_to(root).as_posix(), f.read_text(encoding="utf-8"))]


class Index:
    """Okapi BM25 over the chunks. The section heading is indexed with the text, so a query that
    names a section ("選型理由") finds it even when the body uses other words."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("no documents to index")
        self.chunks = chunks
        self.tfs = [Counter(tokenize(f"{c.section}\n{c.text}")) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def rank(self, query: str, limit: int = PER_QUERY) -> list[tuple[int, float]]:
        """[(chunk index, BM25 score)] best first; chunks sharing no term with the query are dropped."""
        q = set(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lens[i] / self.avg_len)
            for t in q & tf.keys():
                f = tf[t]
                s += self.idf[t] * f * (BM25_K1 + 1) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:limit]


@lru_cache(maxsize=4)
def get_index(root: str = str(ROOT)) -> Index:
    """Built once per process on first use (~50 KB of markdown, a few ms)."""
    return Index(load_corpus(Path(root)))


# ---------------------------------------------------------------- search (template: researcher_graph)

def _check_queries(queries) -> list[str]:
    if not isinstance(queries, list):
        raise TypeError("queries must be a list of strings")
    qs = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if len(qs) != len(queries) or not qs:
        raise ValueError("queries must be 1-3 non-empty strings")
    if len(qs) > MAX_QUERIES:
        raise ValueError(f"at most {MAX_QUERIES} queries per search")
    if any(len(q) > MAX_QUERY_CHARS for q in qs):
        raise ValueError(f"each query must be at most {MAX_QUERY_CHARS} characters")
    return qs


def search(queries: list[str], k: int = TOP_K, index: Index | None = None) -> dict:
    """Rank every query on its own, then fuse with RRF: a passage found by several phrasings of
    the question rises to the top, which is the point of asking the model for diverse queries."""
    qs = _check_queries(queries)
    index = index or get_index()
    fused: dict[int, float] = {}
    for q in qs:
        for rank, (i, _) in enumerate(index.rank(q)):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    best = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
    hits = [{"id": index.chunks[i].id, "source": index.chunks[i].source,
             "section": index.chunks[i].section, "text": index.chunks[i].text,
             "score": round(s, 4)} for i, s in best]
    out = {"queries": qs, "hits": hits, "n_chunks": len(index.chunks)}
    if not hits:
        out["note"] = "文件中沒有找到相關段落"
    return out
