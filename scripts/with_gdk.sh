#!/usr/bin/env bash
set -e
if command -v gdk-pixbuf-query-loaders >/dev/null 2>&1; then
  gdk-pixbuf-query-loaders --update-cache || true
fi
exec "$@"
