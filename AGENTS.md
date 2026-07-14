# AGENTS.md

## General commands

- No setup needed. Do not run "cd", assume you are in the correct path by default.
- Use these commands:
   o `pixi run format`: Apply automatic code formatting
   o `pixi run test`: Run backend tests
   o `pixi run uitest`: Run UI tests
   o `pixi run lint`. Performs linting and static code analysis
   o `pixi run print-untranslated list`: List languages with untranslated strings
   o `pixi run print-untranslated <lang>`: Print untranslated strings from po file

## Code style

- When writing Python, conform to PEP8 with maximum line length of 79 chars
- Keep cyclomatic complexity low. Write small, testable functions
- Never mark your changes with inline comments. Code is for clean, final implementation only
- Retain exiting formatting, docstrings, and comments

## Raygeo (Rust/PyO3 geometry library)

Even though Raygeo is installed as a regular pip dependency, we own it. If the root
cause of an issue is in Raygeo, you should fix it there instead of building a
workaround.
Source repository: https://github.com/barebaric/raygeo

### Testing with a local Raygeo checkout

A `local-raygeo` pixi environment builds raygeo from a local checkout instead
of pulling the PyPI wheel.

```bash
ln -s /path/to/raygeo external/raygeo    # one-time symlink (external/ is gitignored)
pixi install -e local-raygeo
pixi run -e local-raygeo rayforge        # run against local raygeo
pixi run -e local-raygeo test            # test against local raygeo
pixi run -e local-raygeo rebuild-raygeo  # rebuild after editing raygeo source
```

To go back to the PyPI raygeo, just use `pixi run rayforge`.

## Other rules

- Do not run the full test suite prematurely. Fix all linter errors first. Run targeted tests.
- Never use "head" to filter CLI commands! This would hide useful error messages.
- Use proper markdown to put each file into a separate code block.
- File start markers do not belong INTO code blocks. Putting them OUTSIDE is ok.
- Do not make changes unrelated to the current task
- Never remove logging or debugging unless asked by the user
- Do not repeat files unless they have changes
