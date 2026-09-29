# Website documentation

- The user documentation lives in `website/docs`, the blog in `website/blog`
- Translated documentation lives in
  `website/i18n/<lang>/docusaurus-plugin-content-docs/current/`, one directory
  per language under `website/i18n` (list them with `ls website/i18n`). UI
  strings of the website itself are in `website/i18n/<lang>/code.json`
- When a change adds, removes, or alters user-facing behavior, update the
  relevant page in `website/docs` and the matching page in every translated
  directory. Purely internal changes, refactors, and error-handling details do
  not need documentation
- Run `pixi run site-format` after editing website markdown so Prettier
  formatting stays consistent
