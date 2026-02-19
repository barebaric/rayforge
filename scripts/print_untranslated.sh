#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if language code is provided
if [ -z "$1" ]; then
  echo "Error: Language code is required."
  echo "Usage: pixi run print-untranslated <lang>"
  echo "       pixi run print-untranslated list"
  echo "Example: pixi run print-untranslated de"
  echo "         pixi run print-untranslated list"
  exit 1
fi

LANG_CODE="$1"

# List all languages with untranslated strings
if [ "$LANG_CODE" = "list" ]; then
  found=0
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ] && [ -f "$lang_dir/LC_MESSAGES/rayforge.po" ]; then
      if [ "$lang" != "en" ]; then
        PO_FILE="$lang_dir/LC_MESSAGES/rayforge.po"
        if msgattrib --untranslated --no-obsolete "$PO_FILE" 2>/dev/null | grep -q "^msgid"; then
          echo "$lang"
          found=1
        fi
      fi
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "All languages are fully translated."
  fi
  exit 0
fi

# Ignore English language file
if [ "$LANG_CODE" = "en" ]; then
  echo "Warning: English language file should never be translated."
  echo "Available languages:"
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ] && [ -f "$lang_dir/LC_MESSAGES/rayforge.po" ]; then
      if [ "$lang" != "en" ]; then
        echo "  - $lang"
      fi
    fi
  done
  exit 1
fi
PO_FILE="rayforge/locale/${LANG_CODE}/LC_MESSAGES/rayforge.po"

# Check if the .po file exists
if [ ! -f "$PO_FILE" ]; then
  echo "Error: Translation file not found: $PO_FILE"
  echo "Available languages:"
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ] && [ -f "$lang_dir/LC_MESSAGES/rayforge.po" ]; then
      echo "  - $lang"
    fi
  done
  exit 1
fi

# Print untranslated strings (limited to 100 entries by default)
msgattrib --untranslated --no-obsolete "$PO_FILE" | awk '
    /^msgid/ && entry_count >= 100 { exit }
    /^msgid/ { entry_count++ }
    { print }
'
