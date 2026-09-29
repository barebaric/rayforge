# UI strings and translations

Applies to every string that can end up in the UI: labels, tooltips, dialog
headings and bodies, validation and error messages.

## Writing translatable strings

- Import gettext at the top of the file with `from gettext import gettext as _`
  and wrap the text in `_("...")`
- Keep dynamic values out of the message id: translate first, then use named
  placeholders, for example
  `_("URL must start with one of: {schemes}").format(schemes=schemes)`
- Mark such entries as `python-brace-format` in the catalogs and keep the
  placeholder names unchanged in every translation

## Updating the catalogs

- Wrapping a string only marks it for extraction. New messages must also be
  added to the gettext template `rayforge/locale/rayforge.pot` and translated
  in `rayforge/locale/<lang>/LC_MESSAGES/rayforge.po` for every shipped
  language (list them with `ls rayforge/locale`). The `en` catalog keeps empty
  translations, since it falls back to the source text
- Only add or update entries for the strings of your own change; do not
  rewrite unrelated catalog entries

## Avoiding catalog churn

- `pixi run update-translations` rewrites every catalog in the repository,
  including the addon catalogs under `rayforge/builtin_addons/`. Most of those
  files then only differ by a fresh `POT-Creation-Date` and reshuffled entries
  and `#:` source references. Revert every catalog that has no functional
  change, so the review only shows the strings your change really adds
- Keep the catalogs you do have to touch as close to additions-only as
  possible: appending the new entries to the committed file beats regenerating
  it, because a regenerated catalog reorders thousands of unrelated lines and
  buries the actual change. Verify with `git diff` that the catalogs contain
  no deletions, then run `msgfmt --check --check-format` to confirm the result
  is valid and free of duplicate message ids
- To tell a functional change from pure churn, compare a catalog against its
  committed version with ordering and bookkeeping removed:

```bash
git show HEAD:rayforge/locale/de/LC_MESSAGES/rayforge.po \
    | msgcat --sort-output --no-location - > /tmp/old.po
msgcat --sort-output --no-location rayforge/locale/de/LC_MESSAGES/rayforge.po > /tmp/new.po
diff /tmp/old.po /tmp/new.po
```

If the two normalized files are equal, the change is noise and the file should
be reverted.

- Use `pixi run print-untranslated list` and `pixi run print-untranslated
  <lang>` to find missing translations
