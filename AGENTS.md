# AGENTS.md

## Commands

- No setup needed. Do not run "cd", assume you are in the correct path by default.
- `pixi run test`: backend tests; `pixi run uitest`: UI tests
- `pixi run lint`: linting and static analysis; `pixi run format`: format code
- `pixi run site-format`: format website markdown (docs, blog, i18n)
- `pixi run print-untranslated list|<lang>`: find untranslated UI strings
- `python3 scripts/media/gen_image.sh --prompt "..." --out out.png`: generate
  an image with AI (needs pytorch)

## Code style

- Python: PEP8 with a maximum line length of 79 chars. Keep functions small
  and cyclomatic complexity low
- Never mark your changes with inline comments. Code is for clean, final
  implementation only
- Retain existing formatting, docstrings, and comments. When refactoring moves
  code, move the comments and docstrings that belong to it as well
- Always put imports at the top of the file, never inside functions
- Wrap every user-visible string with gettext `_()`; see
  docs/agents/translations.md

## Other rules

- Do not run the full test suite prematurely. Fix all linter errors first.
  Run targeted tests
- Never use "head" to filter CLI commands; it hides useful error messages
- Do not make changes unrelated to the current task
- Never remove logging or debugging unless asked by the user
- In answers, put each file into its own markdown code block. File start
  markers belong outside the block
- Do not repeat files unless they have changes

## Read on demand

These areas are kept out of this file to save context; read the matching file
before working on them:

- docs/agents/translations.md: gettext workflow for UI strings and catalogs
- docs/agents/website.md: website docs, blog, and their translations
- docs/agents/raygeo.md: our Rust/PyO3 geometry library; fix root causes
  there instead of building workarounds
- docs/agents/raydriver.md: our Rust/PyO3 GRBL serial driver
- rayforge/machine/driver/ruidarpa/AGENTS.md: rules for the Ruida driver
