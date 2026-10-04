#!/usr/bin/env python3
"""Estimate the LLM implementation cost of a merged pull request.

Every intermediate commit of the pull request is replayed and the code
lines each commit added or rewrote are counted with scc
(https://github.com/boyter/scc). On top of the tokens for emitting
that code, the model prices the work an LLM session invisibly spends
around it: reading the touched files, replaying the conversation
between turns, the pull request description as the prompt, review
rounds and failed CI runs. Because those overheads rest on
assumptions that vary between tools and tasks, the posted comment
reports a lower bound (pure output cost) alongside a best estimate
(full model) instead of a single number.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_TOKENS_PER_LINE = 10
DEFAULT_PRICE_PER_MTOK = 15.0
DEFAULT_INPUT_RATIO = 15.0
DEFAULT_INPUT_PRICE_PER_MTOK = 3.0
MAX_TABLE_ROWS = 50
MAX_COUNTED_ROUNDS = 20
MAX_ANALYZED_COMMITS = 50
MAX_FILE_READ_BYTES = 20000
BYTES_PER_TOKEN = 4
REVIEW_INPUT_TOKENS = 40000
REVIEW_OUTPUT_TOKENS = 2000
DEBUG_INPUT_TOKENS = 20000
DEBUG_OUTPUT_TOKENS = 1000


def run_git(*args, allow_fail=False):
    proc = subprocess.run(["git", *args], capture_output=True, check=False)
    if proc.returncode != 0 and not allow_fail:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)}: {stderr.strip()}")
    return proc.stdout.decode("utf-8", errors="replace")


def commit_exists(sha):
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True, check=False)
    return proc.returncode == 0


def commit_subject(sha):
    return run_git("log", "-1", "--format=%s", sha).strip()


def commit_parents(sha):
    out = run_git("rev-list", "--parents", "-n", "1", sha).split()
    return out[1:]


def gh_api_json(endpoint):
    proc = subprocess.run(["gh", "api", endpoint],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"warning: GitHub API lookup failed for {endpoint}: "
              f"{proc.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def api_commit_shas(pr_number, repo):
    proc = subprocess.run(
        ["gh", "api", "--paginate",
         f"repos/{repo}/pulls/{pr_number}/commits",
         "--jq", ".[].sha"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"warning: GitHub API lookup failed: {proc.stderr.strip()}",
              file=sys.stderr)
        return []
    return [line.strip()
            for line in proc.stdout.splitlines() if line.strip()]


def fetch_pr_context(pr_number, repo):
    context = {"body": "", "merged_at": "", "reviews": 0, "failed_checks": 0}
    if not pr_number or not repo:
        return context
    data = gh_api_json(f"repos/{repo}/pulls/{pr_number}")
    if data:
        context["body"] = data.get("body") or ""
        context["merged_at"] = data.get("merged_at") or ""
    reviews = gh_api_json(f"repos/{repo}/pulls/{pr_number}/reviews")
    if isinstance(reviews, list):
        context["reviews"] = len(reviews)
    return context


def fetch_failed_checks(shas, repo):
    if not repo:
        return 0
    failures = 0
    for sha in shas[:MAX_ANALYZED_COMMITS]:
        data = gh_api_json(f"repos/{repo}/commits/{sha}/check-runs")
        if not data:
            continue
        failures += sum(1 for run in data.get("check_runs", [])
                        if run.get("conclusion") == "failure")
    return failures


def merge_range_commits(merge_sha):
    parents = commit_parents(merge_sha)
    if len(parents) != 2:
        return []
    return run_git("rev-list", "--reverse",
                   f"{merge_sha}^2", f"^{merge_sha}^1").split()


def head_range_commits(merge_sha, head_sha):
    if not head_sha or not commit_exists(head_sha):
        return []
    base = run_git("merge-base", head_sha, f"{merge_sha}^",
                   allow_fail=True).strip()
    if not base:
        return []
    return run_git("rev-list", "--reverse", f"{base}..{head_sha}").split()


def resolve_commits(args):
    shas = []
    if args.pr_number:
        repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
        if repo:
            shas = [sha for sha in api_commit_shas(args.pr_number, repo)
                    if commit_exists(sha)]
    if not shas:
        shas = merge_range_commits(args.merge_sha)
    if not shas:
        shas = head_range_commits(args.merge_sha, args.head_sha)
    return list(dict.fromkeys(shas))


def extension_of(path):
    suffix = Path(path).suffix.lower()
    return suffix or ".txt"


def parse_commit_diff(sha):
    diff = run_git("diff-tree", "-p", "--root", "--no-commit-id", sha)
    changes = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            current = None
        elif line.startswith("+++ b/"):
            current = line[6:].split("\t")[0]
            changes.setdefault(current, ([], []))
        elif line.startswith("+++ "):
            current = None
        elif current is not None and line.startswith("+"):
            changes[current][0].append(line[1:])
        elif current is not None and line.startswith("-"):
            changes[current][1].append(line[1:])
    return changes


def bucket_by_ext(changes):
    added_by_ext = defaultdict(list)
    deleted_by_ext = defaultdict(list)
    touched = []
    for path, (added, deleted) in changes.items():
        if not added and not deleted:
            continue
        touched.append(path)
        ext = extension_of(path)
        added_by_ext[ext].extend(added)
        deleted_by_ext[ext].extend(deleted)
    return added_by_ext, deleted_by_ext, touched


def write_bucket(directory, ext, lines):
    directory.mkdir(parents=True, exist_ok=True)
    stem = "unknown" if ext == ".txt" else ext[1:]
    target = directory / f"{stem}{ext}"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scc_per_file(scc_bin, directory):
    proc = subprocess.run(
        [scc_bin, "--format", "json", str(directory)],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"scc failed: {proc.stderr.strip()}")
    return {entry["Name"]: entry.get("Code", 0)
            for entry in json.loads(proc.stdout)}


def exploration_tokens(sha, touched):
    total_bytes = 0
    for path in touched:
        proc = subprocess.run(
            ["git", "cat-file", "-s", f"{sha}:{path}"],
            capture_output=True, check=False)
        if proc.returncode != 0:
            continue
        size = int(proc.stdout.decode("utf-8", errors="replace").strip()
                   or 0)
        total_bytes += min(size, MAX_FILE_READ_BYTES)
    return total_bytes // BYTES_PER_TOKEN


def analyze_commit(sha, scc_bin):
    added_by_ext, deleted_by_ext, touched = bucket_by_ext(
        parse_commit_diff(sha))
    counts = {}
    deleted_total = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if added_by_ext:
            added_dir = root / "added"
            for ext, lines in added_by_ext.items():
                write_bucket(added_dir, ext, lines)
            counts = scc_per_file(scc_bin, added_dir)
        if deleted_by_ext:
            deleted_dir = root / "deleted"
            for ext, lines in deleted_by_ext.items():
                write_bucket(deleted_dir, ext, lines)
            deleted_total = sum(scc_per_file(scc_bin, deleted_dir).values())
    read_tokens = exploration_tokens(sha, touched)
    added_total = sum(counts.values())
    print(f"analyzed {sha[:8]}: {added_total} code lines added, "
          f"{deleted_total} rewritten", file=sys.stderr)
    return {"counts": counts, "deleted": deleted_total,
            "read_tokens": read_tokens}


def count_tokens(text):
    return round(len(text.split()) * 1.3)


def token_cost(output_tokens, input_tokens, args):
    return (output_tokens / 1_000_000 * args.price_per_mtok
            + input_tokens / 1_000_000 * args.input_price_per_mtok)


def compute_totals(entries, context, args):
    added = sum(sum(entry["counts"].values()) for entry in entries)
    deleted = sum(entry["deleted"] for entry in entries)
    lines = added + deleted
    output_tokens = round(lines * args.tokens_per_line)
    reviews = min(context["reviews"], MAX_COUNTED_ROUNDS)
    debugs = min(context["failed_checks"], MAX_COUNTED_ROUNDS)
    return {
        "lines": lines,
        "output_tokens": output_tokens,
        "explore": sum(entry["read_tokens"] for entry in entries),
        "replay": round(output_tokens * args.input_ratio),
        "prompt": count_tokens(context["body"]),
        "reviews": reviews,
        "debugs": debugs,
    }


def fmt_int(value):
    return f"{value:,}"


def fmt_money(value):
    return f"${value:,.4f}"


def commit_link(sha):
    server = os.environ.get("GITHUB_SERVER_URL", "").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if server and repo:
        return f"[`{sha[:8]}`]({server}/{repo}/commit/{sha})"
    return f"`{sha[:8]}`"


def table_row(sha, subject, added, deleted, args):
    tokens = round((added + deleted) * args.tokens_per_line)
    cost = token_cost(tokens, 0, args)
    subject = subject.replace("|", "\\|")
    lines = f"{fmt_int(added)} / {fmt_int(deleted)}"
    return (f"| {commit_link(sha)} | {subject} | {lines} "
            f"| {fmt_int(tokens)} | {fmt_money(cost)} |")


def breakdown_table(totals, args):
    components = [
        (f"Code emission ({fmt_int(totals['lines'])} add/rewrite lines)",
         totals["output_tokens"], 0),
        ("Context input (reading touched files)", 0, totals["explore"]),
        (f"Conversation replay (≈{args.input_ratio:g}× output)",
         0, totals["replay"]),
        ("Prompting (pull request description)", 0, totals["prompt"]),
        (f"Review rounds ({totals['reviews']})",
         totals["reviews"] * REVIEW_OUTPUT_TOKENS,
         totals["reviews"] * REVIEW_INPUT_TOKENS),
        (f"Debug cycles ({totals['debugs']} failed checks)",
         totals["debugs"] * DEBUG_OUTPUT_TOKENS,
         totals["debugs"] * DEBUG_INPUT_TOKENS),
    ]
    lines = ["| Component | Est. tokens | Est. cost |",
             "|---|---:|---:|"]
    best_tokens = 0
    best_cost = 0.0
    for label, out_tokens, in_tokens in components:
        if not out_tokens and not in_tokens:
            continue
        cost = token_cost(out_tokens, in_tokens, args)
        best_tokens += out_tokens + in_tokens
        best_cost += cost
        lines.append(f"| {label} | {fmt_int(out_tokens + in_tokens)} "
                     f"| {fmt_money(cost)} |")
    lower_cost = token_cost(totals["output_tokens"], 0, args)
    lines.append(f"| **Lower bound (output only)** "
                 f"| **{fmt_int(totals['output_tokens'])}** "
                 f"| **{fmt_money(lower_cost)}** |")
    lines.append(f"| **Best estimate** | **{fmt_int(best_tokens)}** "
                 f"| **{fmt_money(best_cost)}** |")
    return lines


def model_note(args):
    review_tokens = REVIEW_INPUT_TOKENS + REVIEW_OUTPUT_TOKENS
    debug_tokens = DEBUG_INPUT_TOKENS + DEBUG_OUTPUT_TOKENS
    return (f"Cost model: ≈{args.tokens_per_line:g} tokens per code line "
            f"at ${args.price_per_mtok:g}/Mtok output and "
            f"${args.input_price_per_mtok:g}/Mtok input (assuming heavy "
            f"prompt caching); input ≈ {args.input_ratio:g}× output for "
            f"conversation replay; a review round ≈ "
            f"{fmt_int(review_tokens)} tokens; a failed CI run ≈ "
            f"{fmt_int(debug_tokens)} tokens; file reads capped at "
            f"{MAX_FILE_READ_BYTES // 1000} kB each.")


def parse_iso(text):
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def active_window_note(entries, context):
    merged = parse_iso(context.get("merged_at", ""))
    if not entries or not merged:
        return ""
    first = run_git("log", "-1", "--format=%cI",
                    entries[0]["sha"], allow_fail=True).strip()
    started = parse_iso(first)
    if not started or merged < started:
        return ""
    days = (merged - started).days
    span = "<1 day" if days == 0 else f"{days} day(s)"
    return f"Active window from first commit to merge: {span}."


def build_comment(entries, context, cumulative, args):
    totals = compute_totals(entries, context, args)
    lang_totals = defaultdict(int)
    rows = []
    for entry in entries:
        added = sum(entry["counts"].values())
        for lang, count in entry["counts"].items():
            lang_totals[lang] += count
        rows.append(table_row(entry["sha"], entry["subject"], added,
                              entry["deleted"], args))
    if len(rows) > MAX_TABLE_ROWS:
        hidden = len(rows) - MAX_TABLE_ROWS
        rows = rows[:MAX_TABLE_ROWS]
        rows.append(f"| … | {hidden} more commits "
                    f"(included in totals) | | | |")
    total_tokens = round(totals["lines"] * args.tokens_per_line)
    total_cost = token_cost(total_tokens, 0, args)
    lines_out = [
        "## 🤖 Estimated LLM implementation cost",
        "",
        (f"Replayed each of the **{len(entries)}** commits in this pull "
         f"request and counted the code lines each commit added or "
         f"rewrote:"),
        "",
        ("| Commit | Message | Lines (add/rewrite) | Est. output tokens "
         "| Est. output cost |"),
        "|---|---|---:|---:|---:|",
        *rows,
        (f"| **Total** | | **{fmt_int(totals['lines'])}** | "
         f"**{fmt_int(total_tokens)}** | **{fmt_money(total_cost)}** |"),
        "",
    ]
    if lang_totals:
        languages = ", ".join(
            f"{lang} {fmt_int(count)}"
            for lang, count in sorted(lang_totals.items(),
                                      key=lambda item: -item[1]))
        lines_out += [f"Added lines by language: {languages}", ""]
    lines_out += ["### Full estimate", ""]
    lines_out += breakdown_table(totals, args)
    lines_out += ["", model_note(args), ""]
    window = active_window_note(entries, context)
    if window:
        lines_out += [window, ""]
    if cumulative:
        lines_out += [
            ("Note: this pull request was squash- or rebase-merged, so "
             "its intermediate commits are no longer recoverable; the "
             "estimate is based on the cumulative merge diff."),
            "",
        ]
    lines_out += [
        ("> ⚠️ **Caveat:** the *lower bound* covers only the tokens for "
         "emitting the code. The *best estimate* adds modeled overhead "
         "for exploration, conversation replay, review and debug turns, "
         "but real sessions vary by tool and task — treat it as an "
         "order-of-magnitude figure. Human engineering time is not "
         "included."),
        "",
    ]
    return "\n".join(lines_out)


def cumulative_entry(args):
    entry = analyze_commit(args.merge_sha, args.scc)
    entry["sha"] = args.merge_sha
    entry["subject"] = commit_subject(args.merge_sha)
    return entry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate the LLM implementation cost of a merged "
                    "pull request.")
    parser.add_argument("--merge-sha", required=True,
                        help="merge commit of the pull request")
    parser.add_argument("--head-sha", default="",
                        help="original PR head commit, if available")
    parser.add_argument("--pr-number", type=int, default=0,
                        help="pull request number for the GitHub API")
    parser.add_argument("--repo", default="",
                        help="owner/name repository slug")
    parser.add_argument("--scc", default="scc",
                        help="path to the scc binary")
    parser.add_argument("--out", default="-",
                        help="output file for the markdown comment")
    parser.add_argument(
        "--tokens-per-line", type=float,
        default=float(os.environ.get("SCC_LLM_TOKENS_PER_LINE",
                                     DEFAULT_TOKENS_PER_LINE)))
    parser.add_argument(
        "--price-per-mtok", type=float,
        default=float(os.environ.get("SCC_LLM_PRICE_PER_MTOK",
                                     DEFAULT_PRICE_PER_MTOK)))
    parser.add_argument(
        "--input-ratio", type=float,
        default=float(os.environ.get("SCC_LLM_INPUT_RATIO",
                                     DEFAULT_INPUT_RATIO)),
        help="estimated input tokens as a multiple of output tokens")
    parser.add_argument(
        "--input-price-per-mtok", type=float,
        default=float(os.environ.get("SCC_LLM_INPUT_PRICE_PER_MTOK",
                                     DEFAULT_INPUT_PRICE_PER_MTOK)))
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    shas = resolve_commits(args)
    context = fetch_pr_context(args.pr_number, args.repo)
    context["failed_checks"] = fetch_failed_checks(
        shas or [args.merge_sha], args.repo)
    cumulative = not shas
    if cumulative:
        entries = [cumulative_entry(args)]
    else:
        entries = []
        for sha in shas:
            entry = analyze_commit(sha, args.scc)
            entry["sha"] = sha
            entry["subject"] = commit_subject(sha)
            entries.append(entry)
    comment = build_comment(entries, context, cumulative, args)
    if args.out == "-":
        print(comment)
    else:
        Path(args.out).write_text(comment + "\n", encoding="utf-8")
        print(f"wrote {args.out}: {len(entries)} commits analyzed")


if __name__ == "__main__":
    main()
