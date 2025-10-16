# AGENTS.md

## General commands

- No setup needed. Do not run "cd", assume you are in the correct path by default.
- Use these commands:
   o `pixi run format`. Apply automatic code formatting.
   o `pixi run test`.
   o `pixi run lint`. Performs linting and static code analysis.

## Code style

- When writing Python, conform to PEP8 with maximum line length of 79 chars
- Keep cyclomatic complexity low. Write small, testable functions
- Never mark your changes with inline comments. Code is for clean, final implementation only
- Retain exiting formatting, docstrings, and comments

## Other rules

- File start markers do not belong INTO code blocks. Putting them OUTSIDE is ok.
- Do not make changes unrelated to the current task
- Never remove logging or debugging unless asked by the user
- Do not repeat files unless they have changes
