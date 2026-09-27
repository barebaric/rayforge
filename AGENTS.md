# AGENTS.md

## General commands

- No setup needed. Do not run "cd", assume you are in the correct path by default.
- Use these commands:
   o `pixi run format`: Apply automatic code formatting
   o `pixi run site-format`: Format website markdown files (docs, blog, i18n)
   o `pixi run test`: Run backend tests
   o `pixi run uitest`: Run UI tests
   o `pixi run lint`. Performs linting and static code analysis
   o `pixi run print-untranslated list`: List languages with untranslated strings
   o `pixi run print-untranslated <lang>`: Print untranslated strings from po file
   o `python3 scripts/media/gen_image.sh --prompt "a wooden CNC part" --out part.png`:
      Generates an image using AI. Only works if environment set up for it (pytorch)

## Code style

- When writing Python, conform to PEP8 with maximum line length of 79 chars
- Keep cyclomatic complexity low. Write small, testable functions
- Never mark your changes with inline comments. Code is for clean, final implementation only
- Retain exiting formatting, docstrings, and comments
- When a refactor moves code to another function, class, or file, move the
  comments and docstrings that belong to it as well. Explanations of a
  workaround, an edge case, or a non-obvious reason are part of the code being
  moved and must not be dropped along the way. Only leave a comment behind when
  it describes the old location rather than the moved code, and only delete one
  when the refactor makes it untrue
- Always put imports at the top of the file, never inside functions or methods

## User-visible text and translations

- Every string that can end up in the UI (labels, tooltips, dialog headings and
  bodies, validation and error messages) must be translatable. Import gettext at
  the top of the file with `from gettext import gettext as _` and wrap the text
  in `_("...")`
- Keep dynamic values out of the message id: translate first, then use named
  placeholders, for example
  `_("URL must start with one of: {schemes}").format(schemes=schemes)`. Mark such
  entries as `python-brace-format` in the catalogs and keep the placeholder names
  unchanged in every translation
- Wrapping a string only marks it for extraction. New messages also need to be
  added to the gettext template `rayforge/locale/rayforge.pot` and translated in
  `rayforge/locale/<lang>/LC_MESSAGES/rayforge.po` for the shipped languages
  (`de`, `es`, `fr`, `pt`, `uk`, `zh_CN`). The `en` catalog keeps empty
  translations, since it falls back to the source text
- Only add or update entries for the strings of your own change; do not rewrite
  unrelated catalog entries
- `pixi run update-translations` rewrites every catalog in the repository,
  including the addon catalogs under `rayforge/builtin_addons/`. Most of those
  files then only differ by a fresh `POT-Creation-Date` and reshuffled entries
  and `#:` source references. Revert every catalog that has no functional
  change, so the review only shows the strings your change really adds
- To tell a functional change from pure churn, compare a catalog against its
  committed version with the ordering and bookkeeping removed, for example
  `git show HEAD:<file> | msgcat --sort-output --no-location - > /tmp/old.po`
  and the same for the working copy. If the two normalised files are equal,
  the change is noise and the file should be reverted
- Keep the catalogs you do have to touch as close to additions-only as
  possible. Appending the new entries to the committed file beats regenerating
  it, because a regenerated catalog reorders thousands of unrelated lines and
  buries the actual change. Verify with `git diff` that the catalogs contain no
  deletions, then run `msgfmt --check` to confirm the result is still valid and
  free of duplicate message ids
- Use `pixi run print-untranslated list` and `pixi run print-untranslated <lang>`
  to find missing translations, and `msgfmt --check --check-format` to verify a
  catalog before committing

## Website documentation

- The user documentation lives in `website/docs`, the blog in `website/blog`
- Translated documentation lives in
  `website/i18n/<lang>/docusaurus-plugin-content-docs/current/` for the languages
  `de`, `es`, `fr`, `pt-BR`, `uk`, `zh-CN`. UI strings of the website itself are
  in `website/i18n/<lang>/code.json`
- When a change adds, removes, or alters user-facing behavior, update the
  relevant page in `website/docs` and the matching page in every translated
  directory. Purely internal changes, refactors, and error-handling details do
  not need documentation
- Run `pixi run site-format` after editing website markdown so Prettier
  formatting stays consistent

## Raygeo (Rust/PyO3 geometry library)

Even though Raygeo is installed as a regular pip dependency, we own it. If the root
cause of an issue is in Raygeo, you should fix it there instead of building a
workaround.
Source repository: https://github.com/barebaric/raygeo

### Testing with a local Raygeo checkout

Attention: Running "pixi run", even to lint, reinstalls from PyPi.

`scripts/pixi-raygeo.sh` wraps any pixi command with a
`dependency-override` that uses a local raygeo checkout. The project's
real `pixi.toml`/`pixi.lock` are never permanently modified.

```bash
ln -s /path/to/raygeo external/raygeo    # one-time symlink (external/ is gitignored)
scripts/pixi-raygeo.sh run rayforge      # run against local raygeo
scripts/pixi-raygeo.sh run test          # test against local raygeo
scripts/pixi-raygeo.sh shell             # activate a shell with local raygeo
```

After editing raygeo Rust or Python source, rebuild it with:

```bash
scripts/rebuild-raygeo.sh                # clear uv cache + rebuild raygeo
```

To go back to the PyPI raygeo, just use `pixi run rayforge` without the
wrapper (or any other pixi command).

## Raydriver (Rust/PyO3 GRBL driver)

Raydriver hosts the Rust-native GRBL serial driver, consumed by the
shell driver `GrblSerialNextDriver`
(`rayforge/machine/driver/grbl/serial_next.py`). We also own it.

Source repository: https://github.com/barebaric/raydriver

`python/raydriver/emulator.py` contains a Grbl 1.1 firmware emulator
(not a mock) that the crate's own tests and
`tests/machine/driver/grbl/test_serial_next.py` exercise through the
`MockTransport` test transport. Dialects remain Rayforge data:
command templates are resolved via
`GrblSerialNextDriver._dialect_templates()`.

## Other rules

- Do not run the full test suite prematurely. Fix all linter errors first. Run targeted tests.
- Never use "head" to filter CLI commands! This would hide useful error messages.
- Use proper markdown to put each file into a separate code block.
- File start markers do not belong INTO code blocks. Putting them OUTSIDE is ok.
- Do not make changes unrelated to the current task
- Never remove logging or debugging unless asked by the user
- Do not repeat files unless they have changes

## Addendums
- When working on the Ruida driver, read the AGENTS.md located in the ruidarpa driver directory at rayforge/machine/driver/ruidarpa.
