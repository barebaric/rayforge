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

# Get list of supported languages from main app
SUPPORTED_LANGUAGES=()
for lang_dir in rayforge/locale/*/; do
  lang=$(basename "$lang_dir")
  if [ "$lang" != "rayforge.pot" ] && [ -d "$lang_dir/LC_MESSAGES" ]; then
    SUPPORTED_LANGUAGES+=("$lang")
  fi
done

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

    # Clean up file paths to show relative paths only
    sed -i "s|#: $src_dir/|#: $pkg_name/|g" "$locale_dir/$pkg_name.pot"

    # 2. Create .po files for all supported languages
    echo "  Ensuring .po files exist for all supported languages..."
    for lang in "${SUPPORTED_LANGUAGES[@]}"; do
      lang_dir="$locale_dir/$lang/LC_MESSAGES"
      po_file="$lang_dir/$pkg_name.po"

      if [ ! -d "$lang_dir" ]; then
        echo "    Creating directory $lang_dir..."
        mkdir -p "$lang_dir"
      fi

      if [ ! -f "$po_file" ]; then
        echo "    Initializing $po_file..."
        msginit --no-translator --input="$locale_dir/$pkg_name.pot" --locale="$lang" --output="$po_file" 2>/dev/null || true
        # Set charset to UTF-8 instead of ASCII (msginit defaults to ASCII)
        sed -i 's/charset=ASCII/charset=UTF-8/' "$po_file"
      fi
    done

      # 3. Update existing .po files with msgmerge
      echo " Merging .pot with .po files..."
      for lang_dir in "$locale_dir"/*/; do
        lang=$(basename "$lang_dir")
        if [ -d "$lang_dir/LC_MESSAGES" ]; then
          if [ -f "$lang_dir/LC_MESSAGES/$pkg_name.po" ]; then
            echo "    Updating $lang_dir/LC_MESSAGES/$pkg_name.po"
            msguniq "$lang_dir/LC_MESSAGES/$pkg_name.po" -o "$lang_dir/LC_MESSAGES/$pkg_name.po" 2>/dev/null || true
            msgmerge --update -N "$lang_dir/LC_MESSAGES/$pkg_name.po" "$locale_dir/$pkg_name.pot" 2>/dev/null || true
            msgattrib --no-obsolete --output-file="$lang_dir/LC_MESSAGES/$pkg_name.po" "$lang_dir/LC_MESSAGES/$pkg_name.po" 2>/dev/null || true
            # Ensure charset is UTF-8 instead of ASCII
            sed -i 's/charset=ASCII/charset=UTF-8/' "$lang_dir/LC_MESSAGES/$pkg_name.po"
          fi
        fi
      done
  fi

  # 4. Compile .po files to .mo files
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
  find rayforge/ -name "*.py" \
    -not -path "*/builtin_addons/*" \
    -not -path "*/private_addons/*" | \
    xgettext --from-code=UTF-8 --add-location=file -o rayforge/locale/rayforge.pot -f -

  # 2. Update existing .po files with msgmerge for main app
  echo "Merging .pot with .po files..."
  for lang_dir in rayforge/locale/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/LC_MESSAGES" ]; then
      echo "  Updating $lang_dir/LC_MESSAGES/rayforge.po"
      msguniq "$lang_dir/LC_MESSAGES/rayforge.po" -o "$lang_dir/LC_MESSAGES/rayforge.po" 2>/dev/null || true
      msgmerge --update -N "$lang_dir/LC_MESSAGES/rayforge.po" rayforge/locale/rayforge.pot
      msgattrib --no-obsolete --output-file="$lang_dir/LC_MESSAGES/rayforge.po" "$lang_dir/LC_MESSAGES/rayforge.po" 2>/dev/null || true
      # Ensure charset is UTF-8 instead of ASCII
      sed -i 's/charset=ASCII/charset=UTF-8/' "$lang_dir/LC_MESSAGES/rayforge.po"
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

# 4. Process addons
echo ""
echo "Processing addons..."
for addon_type in builtin_addons private_addons; do
  if [ -d "rayforge/$addon_type" ]; then
    for addon_dir in rayforge/$addon_type/*/; do
      yaml_file="$addon_dir/rayforge-addon.yaml"
      
      if [ ! -f "$yaml_file" ]; then
        continue
      fi
      
      # Extract package name from YAML file
      pkg_name=$(grep -E "^name:" "$yaml_file" | sed 's/name:[[:space:]]*//')
      
      if [ -z "$pkg_name" ]; then
        continue
      fi
      
      src_dir="$addon_dir$pkg_name"
      locale_dir="$addon_dir/locale"
      
      if [ -d "$src_dir" ] && [ -d "$locale_dir" ]; then
        process_package "$pkg_name" "$src_dir" "$locale_dir"
      fi
    done
  fi
done

# Adjust the final message based on the mode.
if [ "$COMPILE_ONLY" = false ]; then
  echo ""
  echo "Translation update complete. Remember to translate new strings in .po files."
else
  echo ""
  echo "Compilation complete."
fi
