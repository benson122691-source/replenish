#!/usr/bin/env python3
"""Run the 30-case assistant evaluation against the real Gemini API and write the report.

Every case is 2-4 model calls, so the full run is ~60-120 requests. On the free tier that can hit
the per-minute or per-day quota: on a 429 the script waits and retries the case, and if the quota
stays exhausted it stops and prints the --only list of cases still to run.
Needs GEMINI_API_KEY. The printed cost is what the paid tier would charge; the free tier is $0.

Usage:
  python scripts/eval_assistant.py                          # all 30 cases -> docs/assistant-eval.md
  python scripts/eval_assistant.py --only s05 r03           # a few cases, report not overwritten
  python scripts/eval_assistant.py --model gemini-3.8-flash # another model (default: GEMINI_MODEL)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replenishai import assistant
from replenishai.assistant_eval import grade
from replenishai.service import load_artifacts

CASES = ROOT / "eval" / "assistant_cases.json"
RAW_OUT = ROOT / "artifacts" / "assistant_eval.json"
REPORT = ROOT / "docs" / "assistant-eval.md"
# USD per 1M tokens, paid tier (ai.google.dev/gemini-api/docs/pricing, 2026-11). Thinking tokens bill as output.
PRICES = {"gemini-2.5-flash": (0.30, 2.50), "gemini-3.8-flash": (0.75, 3.75)}
RETRY_WAITS = (30, 60, 120)              # seconds to wait after a 429 before retrying the same case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--model", default=assistant.MODEL, help="Gemini model id")
    args = ap.parse_args()
    try:
        client = assistant.make_client()
    except assistant.AssistantUnavailable as e:
        sys.exit(f"Cannot run the evaluation: {e}\n"
                 "Create a key at https://aistudio.google.com/apikey, then "
                 "(PowerShell) $env:GEMINI_API_KEY = '...' and rerun.")

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    art = load_artifacts(ROOT / "artifacts")

    rows, raw, tok_in, tok_out = [], [], 0, 0
    for c in cases:
        t0 = time.perf_counter()
        res = None
        for wait in (*RETRY_WAITS, None):
            try:
                res = assistant.ask(c["question"], art, client=client, model=args.model)
                break
            except assistant.RateLimited as e:
                if wait is None:
                    done = {r[0]["id"] for r in rows}
                    left = [x["id"] for x in cases if x["id"] not in done]
                    sys.exit(f"{e}. Probably the daily free quota. Done {len(done)}/{len(cases)} "
                             f"(nothing written). Rerun later with: --only {' '.join(left)}")
                print(f"{c['id']}  429, waiting {wait}s ...")
                time.sleep(wait)
            except assistant.AssistantUnavailable as e:
                sys.exit(f"Cannot run the evaluation: {e}")
        secs = time.perf_counter() - t0
        calls = [{"name": t.name, "input": t.input, "ok": t.ok, "output": t.output} for t in res.tool_calls]
        g = grade(c, res.answer, calls)
        tok_in, tok_out = tok_in + res.input_tokens, tok_out + res.output_tokens
        rows.append((c, g, res, secs))
        raw.append({"case": c, "answer": res.answer, "calls": calls, "served_by": res.served_by,
                    "stop_reason": res.stop_reason, "seconds": round(secs, 1),
                    "tool_ok": g.tool_ok, "grounded": g.grounded,
                    "reasons": g.reasons, "ungrounded": g.ungrounded})
        flag = "PASS" if g.tool_ok else "FAIL"
        print(f"{c['id']}  {flag}  grounded={'yes' if g.grounded else 'NO '}  {secs:5.1f}s  "
              f"{'; '.join(g.reasons + [f'ungrounded {g.ungrounded}'] * (not g.grounded))}")

    n = len(rows)
    acc = sum(g.tool_ok for _, g, _, _ in rows) / n
    grounded = sum(g.grounded for _, g, _, _ in rows) / n
    price_in, price_out = PRICES.get(args.model, (float("nan"), float("nan")))
    cost = tok_in / 1e6 * price_in + tok_out / 1e6 * price_out
    print(f"\ntool-call accuracy {acc:.1%} ({sum(g.tool_ok for _, g, _, _ in rows)}/{n})   "
          f"grounded answers {grounded:.1%}   tokens in={tok_in} out={tok_out}   "
          f"paid-tier cost ≈ ${cost:.2f} (free tier: $0)")

    RAW_OUT.parent.mkdir(exist_ok=True)
    RAW_OUT.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.only:
        return

    by_cat = defaultdict(list)
    for c, g, _, _ in rows:
        by_cat[c["category"]].append(g)
    names = {"query": "查詢", "simulate": "模擬（補貨試算）", "refuse": "拒答／澄清"}
    lines = [
        "# LLM 助理評測結果（自動產生）", "",
        f"日期：{date.today()}｜模型：`{args.model}`（temperature 0）｜"
        f"題目：`eval/assistant_cases.json`（{n} 題）｜原始紀錄：`artifacts/assistant_eval.json`", "",
        "**資料來源：合成資料。** 這份評測量的是「助理有沒有呼叫對的工具、會不會亂編數字」，不是預測準不準。", "",
        f"- **工具呼叫正確率：{acc:.1%}**（{sum(g.tool_ok for _, g, _, _ in rows)}/{n}；成功標準 Q5-4 為 ≥ 80%）",
        f"- 回答中所有數字都能對應到題目或工具輸出：{grounded:.1%}",
        f"- Token：輸入 {tok_in:,}、輸出 {tok_out:,}；以付費方案定價估算約 ${cost:.2f}（免費方案不收費）", "",
        "| 類別 | 題數 | 工具呼叫正確 | 數字可追溯 |", "|:--|--:|--:|--:|",
    ]
    for cat, gs in by_cat.items():
        lines.append(f"| {names.get(cat, cat)} | {len(gs)} | {sum(g.tool_ok for g in gs)} | {sum(g.grounded for g in gs)} |")
    lines += ["", "## 逐題", "", "| 題 | 問題 | 呼叫的工具 | 正確 | 數字 | 備註 |", "|:--|:--|:--|:--|:--|:--|"]
    for c, g, res, _ in rows:
        tools = "、".join(f"{t.name}{'' if t.ok else '（錯誤）'}" for t in res.tool_calls) or "—"
        note = "；".join(g.reasons + ([f"未對應的數字：{', '.join(g.ungrounded)}"] if g.ungrounded else []))
        lines.append(f"| {c['id']} | {c['question']} | {tools} | {'✔' if g.tool_ok else '✘'} | "
                     f"{'✔' if g.grounded else '✘'} | {note.replace('|', '/')} |")
    lines += ["", "評分規則見 `replenishai/assistant_eval.py`。「數字可追溯」容許四捨五入、比率×100、"
              "兩個工具數字的差或和；被標出的數字需要人工確認是否真的是編的。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
