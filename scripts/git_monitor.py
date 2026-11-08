#!/usr/bin/env python3
"""git monitor: compare the real git history against schedule.json.

A milestone counts as DONE when a commit whose subject starts with its tag
(e.g. "[M2] ...") exists. Commit dates are read from git as-is (never edited).

Usage:
    python scripts/git_monitor.py              # status table
    python scripts/git_monitor.py --json       # machine-readable
    python scripts/git_monitor.py --today 2026-11-10   # simulate another day
Exit code is 1 when any milestone is BEHIND (handy for hooks / CI).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent


def read_commits(tz: ZoneInfo) -> list[dict]:
    out = subprocess.run(
        ["git", "log", "--pretty=format:%H|%aI|%s"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    commits = []
    for line in out.stdout.splitlines():
        sha, iso, subject = line.split("|", 2)
        when = datetime.fromisoformat(iso).astimezone(tz)
        commits.append({"sha": sha[:7], "when": when, "subject": subject})
    return commits


def evaluate(schedule: dict, commits: list[dict], today: date) -> list[dict]:
    rows = []
    for m in schedule["milestones"]:
        tag = f"[{m['id']}]"
        hits = [c for c in commits if c["subject"].startswith(tag)]
        done_commit = min(hits, key=lambda c: c["when"]) if hits else None
        due = date.fromisoformat(m["due"])
        missing = [p for p in m.get("deliverables", []) if not (ROOT / p).exists()]
        if done_commit:
            on_time = done_commit["when"].date() <= due
            status = "DONE" if on_time else "DONE (late)"
        elif today > due:
            status = "BEHIND"
        elif today == due:
            status = "DUE TODAY"
        else:
            status = "PENDING"
        rows.append({
            "id": m["id"], "title": m["title"], "due": m["due"], "status": status,
            "commit": done_commit["sha"] if done_commit else "-",
            "committed_on": done_commit["when"].strftime("%m-%d %H:%M") if done_commit else "-",
            "missing_files": missing if not done_commit else [],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--today", help="YYYY-MM-DD, override today's date")
    args = ap.parse_args()

    schedule = json.loads((ROOT / "schedule.json").read_text())
    tz = ZoneInfo(schedule.get("timezone", "UTC"))
    today = date.fromisoformat(args.today) if args.today else datetime.now(tz).date()
    commits = read_commits(tz)
    rows = evaluate(schedule, commits, today)

    if args.json:
        print(json.dumps({"today": str(today), "milestones": rows,
                          "total_commits": len(commits)}, ensure_ascii=False, indent=2))
    else:
        print(f"{schedule['project']} - git monitor  (today {today}, {len(commits)} commits)\n")
        print(f"{'ID':<4} {'Due':<11} {'Status':<12} {'Commit':<8} {'When':<12} Title")
        for r in rows:
            print(f"{r['id']:<4} {r['due']:<11} {r['status']:<12} {r['commit']:<8} "
                  f"{r['committed_on']:<12} {r['title']}")
            if r["missing_files"] and r["status"] in ("BEHIND", "DUE TODAY"):
                print(f"     missing files: {', '.join(r['missing_files'])}")
        per_day = Counter(c["when"].strftime("%m-%d") for c in commits)
        if per_day:
            print("\nCommits per day: " + ", ".join(f"{d}: {n}" for d, n in sorted(per_day.items())))

    return 1 if any(r["status"] == "BEHIND" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
