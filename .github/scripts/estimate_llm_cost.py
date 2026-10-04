#!/usr/bin/env python3
"""Estimate the LLM token cost of implementing a merged pull request.

Every intermediate commit of the pull request is replayed, the code
lines each commit added are counted with scc
(https://github.com/boyter/scc) and converted into an approximate
output token cost. Lines that were added in one commit and rewritten
in a later one are counted once per commit, mirroring the tokens an
LLM implementing the change incrementally would have emitted.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DEFAULT_TOKENS_PER_LINE = 10
DEFAULT_PRICE_PER_MTOK = 15.0
MAX_TABLE_ROWS = 50


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


def added_lines_by_extension(sha):
    diff = run_git("diff-tree", "-p", "--root", "--no-commit-id", sha)
    buckets = defaultdict(list)
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].split("\t")[0]
        elif line.startswith("+++ "):
            current = None
        elif current is not None and line.startswith("+"):
            buckets[extension_of(current)].append(line[1:])
    return buckets


def write_patch_files(directory, buckets):
    for ext, lines in sorted(buckets.items()):
        stem = "unknown" if ext == ".txt" else ext[1:]
        target = directory / f"{stem}{ext}"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scc_code_lines(scc_bin, directory):
    proc = subprocess.run(
        [scc_bin, "--format", "json", str(directory)],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"scc failed: {proc.stderr.strip()}")
    return {entry["Name"]: entry.get("Code", 0)
            for entry in json.loads(proc.stdout) if entry.get("Code")}


def analyze_commit(sha, scc_bin):
    buckets = added_lines_by_extension(sha)
    if not buckets:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        write_patch_files(Path(tmp), buckets)
        counts = scc_code_lines(scc_bin, Path(tmp))
    total = sum(counts.values())
    print(f"analyzed {sha[:8]}: {total} code lines added", file=sys.stderr)
    return counts


def cost_of(lines, args):
    tokens = round(lines * args.tokens_per_line)
    return tokens, tokens / 1_000_000 * args.price_per_mtok


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


def table_row(sha, subject, lines, args):
    tokens, cost = cost_of(lines, args)
    subject = subject.replace("|", "\\|")
    return (f"| {commit_link(sha)} | {subject} | {fmt_int(lines)} | "
            f"{fmt_int(tokens)} | {fmt_money(cost)} |")


def build_comment(entries, cumulative, args):
    lang_totals = defaultdict(int)
    total_lines = 0
    rows = []
    for entry in entries:
        lines = sum(entry["counts"].values())
        for lang, count in entry["counts"].items():
            lang_totals[lang] += count
        total_lines += lines
        rows.append(table_row(entry["sha"], entry["subject"], lines, args))
    if len(rows) > MAX_TABLE_ROWS:
        hidden = len(rows) - MAX_TABLE_ROWS
        rows = rows[:MAX_TABLE_ROWS]
        rows.append(f"| … | {hidden} more commits "
                    f"(included in totals) | | | |")
    total_tokens, total_cost = cost_of(total_lines, args)
    languages = ", ".join(
        f"{lang} {fmt_int(count)}"
        for lang, count in sorted(lang_totals.items(),
                                  key=lambda item: -item[1]))
    lines_out = [
        "## 🤖 Estimated LLM implementation cost",
        "",
        (f"Replayed each of the **{len(entries)}** commits in this pull "
         f"request and counted the code lines each commit added:"),
        "",
        ("| Commit | Message | Code lines added | Est. tokens | "
         "Est. cost |"),
        "|---|---|---:|---:|---:|",
        *rows,
        (f"| **Total** | | **{fmt_int(total_lines)}** | "
         f"**{fmt_int(total_tokens)}** | **{fmt_money(total_cost)}** |"),
        "",
    ]
    if languages:
        lines_out += [f"Lines by language: {languages}", ""]
    lines_out += [
        (f"Cost model: ≈{args.tokens_per_line:g} tokens per code line at "
         f"${args.price_per_mtok:g} per 1M output tokens."),
        "",
    ]
    if cumulative:
        lines_out += [
            ("Note: this pull request was squash- or rebase-merged, so "
             "its intermediate commits are no longer recoverable; the "
             "estimate is based on the cumulative merge diff."),
            "",
        ]
    lines_out += [
        ("> ⚠️ **Caveat:** this is a *pure token-cost* estimate — the "
         "price of generating the added lines as output tokens. It does "
         "not account for engineering effort, exploration, prompting, "
         "review or debugging."),
        "",
    ]
    return "\n".join(lines_out)


def cumulative_entry(args):
    sha = args.merge_sha
    return {"sha": sha, "subject": commit_subject(sha),
            "counts": analyze_commit(sha, args.scc)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate the LLM token cost of a merged pull request.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    shas = resolve_commits(args)
    cumulative = not shas
    if cumulative:
        entries = [cumulative_entry(args)]
    else:
        entries = [{"sha": sha, "subject": commit_subject(sha),
                    "counts": analyze_commit(sha, args.scc)}
                   for sha in shas]
    comment = build_comment(entries, cumulative, args)
    if args.out == "-":
        print(comment)
    else:
        Path(args.out).write_text(comment + "\n", encoding="utf-8")
        print(f"wrote {args.out}: {len(entries)} commits analyzed")


if __name__ == "__main__":
    main()
