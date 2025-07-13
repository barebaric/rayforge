#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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

# 3. Compile .po files to .mo files
echo "Compiling .mo files..."
for lang_dir in rayforge/locale/*/; do
  lang=$(basename "$lang_dir")
  if [ -d "$lang_dir/LC_MESSAGES" ]; then
    echo "  Compiling $lang_dir/LC_MESSAGES/rayforge.mo"
    msgfmt "$lang_dir/LC_MESSAGES/rayforge.po" -o "$lang_dir/LC_MESSAGES/rayforge.mo"
  fi
done

echo "Translation update complete. Remember to translate new strings in .po files."
