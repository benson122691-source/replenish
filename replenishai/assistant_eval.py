"""Grading for the assistant evaluation (eval/assistant_cases.json). Pure functions, no API calls.

Two separate scores, because they answer different questions:
  tool_ok   - did the model call the right tools with the right arguments, avoid calls it must
              not make, and say what it had to say? (project-brief Q5-4: >= 80%)
  grounded  - is every number in the answer traceable to the question or a tool output
              (allowing rounding, x100 for rates, and differences between tool numbers)?
              This is the "does it make numbers up" check; flagged numbers are listed for review.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations

ID_RE = re.compile(r"\b[A-Z]+_\d+_\d+\b|\b[A-Z]{2}_\d+\b")       # FOODS_1_001, CA_1
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}\b|\b\d{2}-\d{2}\b")
NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
SMALL_INTS = set(range(0, 11)) | {28}            # counts like "3 個原點", "28 天" need no source


@dataclass
class Graded:
    id: str
    category: str
    tool_ok: bool
    grounded: bool
    reasons: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def _same(a, b) -> bool:
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return isinstance(a, (int, float)) and abs(float(a) - float(b)) < 1e-9
    if isinstance(b, str):
        return isinstance(a, str) and a.strip().upper() == b.strip().upper()
    return a == b


def _matches(call: dict, exp: dict) -> bool:
    return call["name"] == exp["tool"] and all(
        k in call["input"] and _same(call["input"][k], v) for k, v in exp["args"].items())


def _numbers_in(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers_in(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers_in(v)]
    if isinstance(obj, str):
        return [float(m.replace(",", "")) for m in NUM_RE.findall(ID_RE.sub(" ", obj))]
    return []


def allowed_numbers(question: str, tool_outputs: list[str]) -> list[float]:
    base = _numbers_in(question)
    tool_nums: list[float] = []
    for out in tool_outputs:
        try:
            tool_nums += _numbers_in(json.loads(out))
        except json.JSONDecodeError:
            tool_nums += _numbers_in(out)
    vals = set(abs(x) for x in base + tool_nums)
    vals |= {abs(x) * 100 for x in tool_nums if abs(x) <= 1.5}       # fill rate 0.977 -> 97.7%
    small = sorted({round(abs(x), 4) for x in tool_nums})
    if len(small) <= 400:                                            # what-if deltas, sums of two
        for x, y in combinations(small, 2):
            vals.add(abs(x - y))
            vals.add(x + y)
    return sorted(vals)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def ungrounded_numbers(answer: str, question: str, tool_outputs: list[str]) -> list[str]:
    text = DATE_RE.sub(" ", ID_RE.sub(" ", answer))
    allowed = allowed_numbers(question, tool_outputs)
    bad = []
    for tok in NUM_RE.findall(text):
        n = float(tok.replace(",", ""))
        if n in SMALL_INTS:
            continue
        tol = 0.5 * 10 ** (-_decimals(tok)) + 1e-9                  # n may be a rounding of v
        if not any(abs(n - v) <= tol for v in allowed):
            bad.append(tok)
    return bad


def grade(case: dict, answer: str, calls: list[dict]) -> Graded:
    """calls: [{"name", "input", "ok", "output"}] in the order the model made them."""
    reasons = []
    for exp in case.get("expect", []):
        if not any(_matches(c, exp) for c in calls):
            reasons.append(f"missing call {exp['tool']}{json.dumps(exp['args'], ensure_ascii=False)}")
    for c in calls:
        if c["ok"] and c["name"] in case.get("forbid_ok", []):
            reasons.append(f"forbidden successful call {c['name']}{json.dumps(c['input'], ensure_ascii=False)}")
    if case.get("answer_any") and not any(s in answer for s in case["answer_any"]):
        reasons.append(f"answer lacks any of {case['answer_any']}")
    # sources: what the tools returned AND the arguments the model sent (e.g. default 0.95)
    sources = [s for c in calls if c["ok"] for s in (c["output"], json.dumps(c["input"]))]
    bad = ungrounded_numbers(answer, case["question"], sources)
    return Graded(case["id"], case["category"], tool_ok=not reasons, grounded=not bad,
                  reasons=reasons, ungrounded=bad)
