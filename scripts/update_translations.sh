#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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
  find rayforge/ -name "*.py" | xgettext --from-code=UTF-8 -o rayforge/locale/rayforge.pot -f -

  # 2. Update existing .po files with msgmerge
  echo "Merging .pot with .po files..."
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ]; then
      echo "  Updating $lang_dir/LC_MESSAGES/rayforge.po"
      msgmerge --update "$lang_dir/LC_MESSAGES/rayforge.po" rayforge/locale/rayforge.pot
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
