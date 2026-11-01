#!/usr/bin/env bash
# Fails if any Mach-O file in the app bundle declares a minimum macOS
# version (LC_BUILD_VERSION.minos or LC_VERSION_MIN_MACOSX.version)
# above the given threshold. Binaries without a version load command
# are not flagged; dyld treats them as unconstrained.
#
# Usage: check_minos.sh [max_version] [app_bundle]
set -euo pipefail

MAX_VERSION="${1:-12.0}"
APP_PATH="${2:-dist/Rayforge.app}"

if [ ! -d "$APP_PATH" ]; then
    echo "App bundle not found at $APP_PATH" >&2
    exit 1
fi

version_le() {
    local a="$1" b="$2" first
    [ "$a" = "$b" ] && return 0
    first=$(printf '%s\n%s\n' "$a" "$b" | \
        sort -t. -k1,1n -k2,2n -k3,3n | head -n 1)
    [ "$first" = "$a" ]
}

fail=0
scanned=0
while IFS= read -r -d '' f; do
    file -b "$f" | grep -q "Mach-O" || continue
    scanned=$((scanned + 1))
    minos=$(otool -l "$f" 2>/dev/null | awk '
        /^ *cmd LC_BUILD_VERSION$/ { mode = "minos"; next }
        /^ *cmd LC_VERSION_MIN_MACOSX$/ { mode = "version"; next }
        /^ *cmd / { mode = ""; next }
        mode == "minos" && /^ *minos / { print $2 }
        mode == "version" && /^ *version / { print $2 }
    ' | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)
    [ -z "$minos" ] && continue
    if ! version_le "$minos" "$MAX_VERSION"; then
        echo "ERROR: $f requires macOS $minos (max $MAX_VERSION)" >&2
        fail=1
    fi
done < <(find "$APP_PATH" -type f -print0)

if [ "$fail" -eq 0 ]; then
    echo "OK: $scanned Mach-O files target macOS <= $MAX_VERSION"
fi
exit "$fail"
