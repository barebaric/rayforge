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

# Function to process translations for a package
process_package() {
  local pkg_name="$1"
  local src_dir="$2"
  local locale_dir="$3"

  if [ ! -d "$src_dir" ]; then
    return
  fi

  if [ "$COMPILE_ONLY" = false ]; then
    echo "Updating translations for $pkg_name..."

    # 1. Extract new strings to .pot file
    echo "  Extracting strings to $locale_dir/$pkg_name.pot..."
    find "$src_dir" -name "*.py" | xgettext --from-code=UTF-8 --add-location=file -o "$locale_dir/$pkg_name.pot" -f - 2>/dev/null || true

    # 2. Update existing .po files with msgmerge
    echo "  Merging .pot with .po files..."
    for lang_dir in "$locale_dir"/*/; do
      lang=$(basename "$lang_dir")
      if [ -d "$lang_dir/LC_MESSAGES" ]; then
        if [ -f "$lang_dir/LC_MESSAGES/$pkg_name.po" ]; then
          echo "    Updating $lang_dir/LC_MESSAGES/$pkg_name.po"
          msgmerge --update -N "$lang_dir/LC_MESSAGES/$pkg_name.po" "$locale_dir/$pkg_name.pot" 2>/dev/null || true
          msgattrib --no-obsolete --output-file="$lang_dir/LC_MESSAGES/$pkg_name.po" "$lang_dir/LC_MESSAGES/$pkg_name.po" 2>/dev/null || true
        fi
      fi
    done
  fi

  # 3. Compile .po files to .mo files
  echo "  Compiling .mo files..."
  for lang_dir in "$locale_dir"/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ]; then
      if [ -f "$lang_dir/LC_MESSAGES/$pkg_name.po" ]; then
        echo "    Compiling $lang_dir/LC_MESSAGES/$pkg_name.mo"
        msgfmt "$lang_dir/LC_MESSAGES/$pkg_name.po" -o "$lang_dir/LC_MESSAGES/$pkg_name.mo"
      fi
    fi
  done
}

# Steps 1 and 2 are skipped if --compile-only is passed.
if [ "$COMPILE_ONLY" = false ]; then
  echo "Updating translation files..."

  # 1. Extract new strings to .pot file for main app
  echo "Extracting strings to rayforge/locale/rayforge.pot..."
  find rayforge/ -name "*.py" -not -path "*/builtin_packages/*" | xgettext --from-code=UTF-8 --add-location=file -o rayforge/locale/rayforge.pot -f -

  # 2. Update existing .po files with msgmerge for main app
  echo "Merging .pot with .po files..."
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ]; then
      echo "  Updating $lang_dir/LC_MESSAGES/rayforge.po"
      msgmerge --update -N "$lang_dir/LC_MESSAGES/rayforge.po" rayforge/locale/rayforge.pot
      msgattrib --no-obsolete --output-file="$lang_dir/LC_MESSAGES/rayforge.po" "$lang_dir/LC_MESSAGES/rayforge.po"
    fi
  done
else
  echo "Compile-only mode: Skipping .pot extraction and .po update."
fi

# 3. Compile .po files to .mo files for main app
echo "Compiling .mo files..."
for lang_dir in rayforge/locale/*/; do
  lang=$(basename "$lang_dir")
  if [ -d "$lang_dir/LC_MESSAGES" ]; then
    echo "  Compiling $lang_dir/LC_MESSAGES/rayforge.mo"
    msgfmt "$lang_dir/LC_MESSAGES/rayforge.po" -o "$lang_dir/LC_MESSAGES/rayforge.mo"
  fi
done

# 4. Process builtin packages
echo ""
echo "Processing builtin packages..."
for pkg_dir in rayforge/builtin_packages/*/; do
  pkg_name=$(basename "$pkg_dir")
  src_dir="$pkg_dir"
  locale_dir="$pkg_dir/locales"

  # Skip if no locales directory exists
  if [ ! -d "$locale_dir" ]; then
    continue
  fi

  process_package "$pkg_name" "$src_dir" "$locale_dir"
done

# Adjust the final message based on the mode.
if [ "$COMPILE_ONLY" = false ]; then
  echo ""
  echo "Translation update complete. Remember to translate new strings in .po files."
else
  echo ""
  echo "Compilation complete."
fi
