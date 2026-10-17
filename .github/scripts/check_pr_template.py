#!/usr/bin/env python3
"""Check that a pull request description follows the PR template.

Pull requests that leave template sections empty or the checklist
unticked are closed automatically with a comment listing what is
missing. Editing the description into compliance reopens the pull
request. Drafts, bots, and users with write access to the repository
are exempt. Only pull requests opened after the check was introduced
are subject to it.
"""

import argparse
import json
import os
import re
import subprocess

MARKER = "<!-- pr-template-check-bot -->"
LABEL = "invalid-template"
MIN_ANSWER_CHARS = 30
ENFORCED_SINCE = "2026-10-15T11:12:21Z"

BOT_LOGINS = ("dependabot[bot]", "github-actions[bot]")

HEADING_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$", re.MULTILINE)

SECTION_HEADINGS = (
    "What does this PR change?",
    "Why did you choose this specific architectural approach?",
    "How did you manually test this locally?",
    "Checklist",
)

SECTION_PLACEHOLDERS = {
    "What does this PR change?": (
        (
            "A clear and concise description of the change and the "
            "problem it solves."
        ),
    ),
    "Why did you choose this specific architectural approach?": (
        (
            "Explain your reasoning in terms of Rayforge's existing "
            "architecture. Which alternatives did you consider, and why "
            "did you reject them?"
        ),
    ),
    "How did you manually test this locally?": (
        (
            "Describe the exact steps you performed to verify the change, "
            "so a maintainer can reproduce them:"
        ),
        "1. 2. 3.",
    ),
}

CHECKBOX_STATEMENTS = (
    "I am not an AI. I have manually read the code and understand it.",
    (
        "I take full responsibility for the quality and architectural "
        "fit of this code."
    ),
    "I agree to publish this code under the MIT license.",
)


def gh_api(*args):
    proc = subprocess.run(
        ["gh", "api", *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {args}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def gh_run(*args, stdin=None):
    proc = subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh {args}: {proc.stderr.strip()}")
    return proc.stdout


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()


def split_sections(body):
    matches = list(HEADING_RE.finditer(body))
    sections = {}
    for index, match in enumerate(matches):
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(body)
        heading = normalize(match.group(1))
        sections[heading] = normalize(body[match.end() : end])
    return sections


def answer_text(text, heading):
    for placeholder in SECTION_PLACEHOLDERS.get(heading, ()):
        text = text.replace(normalize(placeholder), "")
    return text.strip()


def check_checklist(text):
    problems = []
    for statement in CHECKBOX_STATEMENTS:
        pattern = re.compile(
            r"[-*+] \[([x ])\] " + re.escape(statement), re.IGNORECASE
        )
        match = pattern.search(text)
        if not match:
            problems.append(f'checklist item "{statement}" is missing')
        elif match.group(1).lower() != "x":
            problems.append(f'checklist item "{statement}" is not ticked')
    return problems


def find_problems(body):
    sections = split_sections(body)
    problems = []
    for heading in SECTION_HEADINGS:
        if heading not in sections:
            problems.append(f'section "**{heading}**" is missing')
            continue
        if heading == "Checklist":
            problems.extend(check_checklist(sections[heading]))
            continue
        answer = answer_text(sections[heading], heading)
        if len(answer) < MIN_ANSWER_CHARS:
            problems.append(f'section "**{heading}**" is not filled out')
    return problems


def rejection_comment(repo, problems):
    template = (
        f"https://github.com/{repo}/blob/main/.github/PULL_REQUEST_TEMPLATE.md"
    )
    contributing = f"https://github.com/{repo}/blob/main/CONTRIBUTING.md"
    items = "\n".join(f"- {problem}" for problem in problems)
    return (
        f"{MARKER}\n\n"
        "This pull request was closed automatically because its "
        "description does not follow the "
        f"[pull request template]({template}). Problems found:\n\n"
        f"{items}\n\n"
        "Please edit the pull request description to answer every "
        "section and to tick every checkbox; as soon as it complies, "
        "the pull request is reopened automatically. See "
        f"[CONTRIBUTING.md]({contributing})."
    )


def reopen_comment():
    return (
        f"{MARKER}\n\n"
        "The description now follows the pull request template, so "
        "this pull request was reopened automatically."
    )


def comment_exists(repo, number, body):
    comments = gh_api("--paginate", f"repos/{repo}/issues/{number}/comments")
    return any(c.get("body") == body for c in comments)


def has_write_access(repo, login):
    response = gh_api(f"repos/{repo}/collaborators/{login}/permission")
    return response.get("permission") in ("admin", "write")


def has_label(pr, name):
    return any(l.get("name") == name for l in pr.get("labels", []))


def is_enforced(pr):
    return pr.get("created_at", "") >= ENFORCED_SINCE


def reject(repo, number, problems):
    comment = rejection_comment(repo, problems)
    if not comment_exists(repo, number, comment):
        gh_run("pr", "comment", str(number), "--body-file", "-", stdin=comment)
    gh_run("pr", "close", str(number))
    gh_run(
        "label",
        "create",
        LABEL,
        "--force",
        "--color",
        "E99695",
        "--description",
        "Does not follow the pull request template",
    )
    gh_run("pr", "edit", str(number), "--add-label", LABEL)


def unlabel_and_reopen(repo, number, pr):
    gh_run("pr", "edit", str(number), "--remove-label", LABEL)
    if pr["state"] == "closed" and not pr.get("merged_at"):
        gh_run("pr", "reopen", str(number))
        gh_run(
            "pr",
            "comment",
            str(number),
            "--body-file",
            "-",
            stdin=reopen_comment(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    args = parser.parse_args()

    repo = os.environ["GITHUB_REPOSITORY"]
    pr = gh_api(f"repos/{repo}/pulls/{args.pr_number}")
    if pr.get("draft"):
        return
    login = pr.get("user", {}).get("login", "")
    if login in BOT_LOGINS or has_write_access(repo, login):
        return

    problems = find_problems(pr.get("body") or "")
    if problems:
        if is_enforced(pr) and pr["state"] == "open":
            reject(repo, args.pr_number, problems)
        return
    if has_label(pr, LABEL):
        unlabel_and_reopen(repo, args.pr_number, pr)


if __name__ == "__main__":
    main()
