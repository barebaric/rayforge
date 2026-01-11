#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check gettext version
GETTEXT_VERSION=$(gettext --version | head -n1 | awk '{print $NF}')
REQUIRED_VERSION="0.25"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GETTEXT_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
  echo "Error: gettext version $GETTEXT_VERSION is too old. Version $REQUIRED_VERSION or higher is required."
  exit 1
fi

# Parse arguments
COMPILE_ONLY=false
if [[ "$1" == "--compile-only" ]]; then
  COMPILE_ONLY=true
fi

# Steps 1 and 2 are skipped if --compile-only is passed.
if [ "$COMPILE_ONLY" = false ]; then
  echo "Updating translation files..."

  # 1. Extract new strings to .pot file
  echo "Extracting strings to rayforge/locale/rayforge.pot..."
  find rayforge/ -name "*.py" | xgettext --from-code=UTF-8 --add-location=file -o rayforge/locale/rayforge.pot -f -

  # 2. Update existing .po files with msgmerge
  echo "Merging .pot with .po files..."
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ]; then
      echo "  Updating $lang_dir/LC_MESSAGES/rayforge.po"
      msgmerge --update -N "$lang_dir/LC_MESSAGES/rayforge.po" rayforge/locale/rayforge.pot
    fi
  done
else
  echo "Compile-only mode: Skipping .pot extraction and .po update."
fi


# 3. Compile .po files to .mo files (this step always runs)
echo "Compiling .mo files..."
for lang_dir in rayforge/locale/*/; do
  lang=$(basename "$lang_dir")
  if [ -d "$lang_dir/LC_MESSAGES" ]; then
    echo "  Compiling $lang_dir/LC_MESSAGES/rayforge.mo"
    msgfmt "$lang_dir/LC_MESSAGES/rayforge.po" -o "$lang_dir/LC_MESSAGES/rayforge.mo"
  fi
done

# Adjust the final message based on the mode.
if [ "$COMPILE_ONLY" = false ]; then
  echo "Translation update complete. Remember to translate new strings in .po files."
else
  echo "Compilation complete."
fi
