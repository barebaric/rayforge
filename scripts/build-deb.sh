#!/bin/bash
set -e

# --- Setup & Cleanup ---
BUILD_DIR=$(mktemp -d)
ORIG_DIR=$(pwd)
cleanup() {
    echo "--- Cleaning up temporary build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
    echo "Cleanup complete."
}
trap cleanup EXIT

# --- 1. Dynamic Version Detection ---
echo "--- Determining version from Git repository ---"
if [[ -n "$GITHUB_REF_NAME" && "$GITHUB_REF_TYPE" == "tag" ]]; then
    UPSTREAM_VERSION_RAW="$GITHUB_REF_NAME"
else
    UPSTREAM_VERSION_RAW=$(git describe --tags --always --long | sed -e 's/^v//' -e 's/\([^-]*\)-\([0-9]*\)-g\([0-9a-f]*\)/\1~dev\2~\3/')
fi
UPSTREAM_VERSION="${UPSTREAM_VERSION_RAW#v}"
echo "Detected upstream version: ${UPSTREAM_VERSION}"

# --- 2. Vendor Dependencies: Pre-download wheels ---
echo "--- Vendoring pre-built wheels ---"
TMP_SRC_DIR="${BUILD_DIR}/rayforge-${UPSTREAM_VERSION}"
mkdir -p "${TMP_SRC_DIR}/vendor/sdist"

REQUIREMENTS_FILE="debian/requirements-bundle.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "::error::File not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Use pip download instead of curl/jq to ensure ABI compatibility
# This grabs wheels matching the current system (Ubuntu 24.04/Py3.12)
# which matches both the runner and the PPA builder.
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" != *"=="* ]]; then
        echo "::error::Invalid requirement format (must contain '=='): $line"
        exit 1
    fi
    
    echo "Downloading artifact for $line..."
    # --prefer-binary: Get wheels (prevents compilation on Launchpad)
    # --no-deps: Only vendor the specific packages listed
    python3 -m pip download \
        --dest "${TMP_SRC_DIR}/vendor/sdist" \
        --no-deps \
        --prefer-binary \
        "$line"

done < "$REQUIREMENTS_FILE"

if [[ -z "$(ls -A "${TMP_SRC_DIR}/vendor/sdist/" 2>/dev/null)" ]]; then
    echo "::error::No wheels were downloaded."
    exit 1
fi

# --- 3. Create Upstream Tarball ---
echo "--- Creating upstream tarball with vendored wheels ---"
rsync -a \
    --exclude='.git' \
    --exclude='.pixi' \
    --exclude='.venv' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='repo' \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='debian' \
    "$ORIG_DIR"/ "$TMP_SRC_DIR"/

TARBALL_NAME="rayforge_${UPSTREAM_VERSION}.orig.tar.gz"
tar -czf "$BUILD_DIR/$TARBALL_NAME" -C "$BUILD_DIR" "rayforge-${UPSTREAM_VERSION}"
echo "Created: $BUILD_DIR/$TARBALL_NAME"

# --- 4. Build the Package ---
cd "$BUILD_DIR"
cp -r "$ORIG_DIR/debian" "$TMP_SRC_DIR/"
cd "$TMP_SRC_DIR"

MAINTAINER_INFO=$(grep '^Maintainer:' debian/control | head -n 1 | sed 's/Maintainer: //')
export DEBEMAIL=$(echo "$MAINTAINER_INFO" | sed -E 's/.*<(.*)>.*/\1/')
export DEBFULLNAME=$(echo "$MAINTAINER_INFO" | sed -E 's/ <.*//')

# Set the version string based on whether --source is passed (for PPA) or not (for local testing)
if [[ "${1:-}" == "--source" ]]; then
# Use the TARGET_DISTRIBUTION from the environment, defaulting to 'noble' if not set
    TARGET_DIST="${TARGET_DISTRIBUTION:-noble}"
    dch --newversion "${UPSTREAM_VERSION}-1~ppa1~${TARGET_DIST}1" --distribution "$TARGET_DIST" "New PPA release for ${TARGET_DIST}."
else
    dch --newversion "${UPSTREAM_VERSION}-1~local1" "New local build ${UPSTREAM_VERSION}."
fi

# --- 4a. Build Source Package (Strictly source-only for PPA) ---
# Must be built BEFORE binary build to avoid 'debian/files' pollution
echo "--- Building Source Package (for PPA) ---"
env -i \
    HOME="$HOME" \
    PATH="/usr/sbin:/usr/bin:/sbin:/bin" \
    DEBEMAIL="$DEBEMAIL" \
    DEBFULLNAME="$DEBFULLNAME" \
    dpkg-buildpackage -S -sa -us -uc

# --- 4b. Build Binary Package (For local testing) ---
echo "--- Building Binary Package (for testing) ---"
env -i \
    HOME="$HOME" \
    PATH="/usr/sbin:/usr/bin:/sbin:/bin" \
    DEBEMAIL="$DEBEMAIL" \
    DEBFULLNAME="$DEBFULLNAME" \
    dpkg-buildpackage -b -us -uc

# --- 5. Copy Artifacts ---
echo "--- Copying build artifacts back to project's dist/ directory ---"
mkdir -p "$ORIG_DIR/dist"
# This finds the .deb, .dsc, .tar.gz, and the new _source.changes
find "$BUILD_DIR" -maxdepth 1 -name 'rayforge*' -type f -exec cp -v {} "$ORIG_DIR/dist/" \;

echo "Build complete. Artifacts are in the dist/ directory."
