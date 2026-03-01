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

# Function to check untranslated strings in a package
check_package() {
  local pkg_name="$1"
  local locale_dir="$2"
  local po_file="$locale_dir/$LANG_CODE/LC_MESSAGES/$pkg_name.po"

  if [ ! -f "$po_file" ]; then
    return
  fi

  if msgattrib --untranslated --no-obsolete "$po_file" 2>/dev/null | grep -q "^msgid"; then
    echo ""
    echo "=== $pkg_name ==="
    msgattrib --untranslated --no-obsolete "$po_file" | awk '
        /^msgid/ && entry_count >= 100 { exit }
        /^msgid/ { entry_count++ }
        { print }
    '
  fi
}

# List all languages with untranslated strings
if [ "$LANG_CODE" = "list" ]; then
  found=0

  # Check main app
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

  # Check builtin packages
  for pkg_dir in rayforge/builtin_packages/*/; do
    pkg_name=$(basename "$pkg_dir")
    locale_dir="$pkg_dir/locales"
    if [ -d "$locale_dir" ]; then
      for lang_dir in "$locale_dir"/*/; do
        lang=$(basename "$lang_dir")
        if [ -d "$lang_dir/LC_MESSAGES" ] && [ -f "$lang_dir/LC_MESSAGES/$pkg_name.po" ]; then
          if [ "$lang" != "en" ]; then
            PO_FILE="$lang_dir/LC_MESSAGES/$pkg_name.po"
            if msgattrib --untranslated --no-obsolete "$PO_FILE" 2>/dev/null | grep -q "^msgid"; then
              echo "$lang ($pkg_name)"
              found=1
            fi
          fi
        fi
      done
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

# Print untranslated strings for main app (limited to 100 entries by default)
echo "=== rayforge (main app) ==="
msgattrib --untranslated --no-obsolete "$PO_FILE" | awk '
    /^msgid/ && entry_count >= 100 { exit }
    /^msgid/ { entry_count++ }
    { print }
'

# Check builtin packages
for pkg_dir in rayforge/builtin_packages/*/; do
  pkg_name=$(basename "$pkg_dir")
  locale_dir="$pkg_dir/locales"
  check_package "$pkg_name" "$locale_dir"
done
