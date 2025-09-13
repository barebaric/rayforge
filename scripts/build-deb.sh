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
# IMPORTANT: Wheels are downloaded to a 'vendor/' directory, NOT 'debian/'.
# This treats them as part of the upstream source, which is the correct approach.
TMP_SRC_DIR="${BUILD_DIR}/rayforge-${UPSTREAM_VERSION}"
mkdir -p "${TMP_SRC_DIR}/vendor/sdist"

REQUIREMENTS_FILE="debian/requirements-bundle.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "::error::File not found: $REQUIREMENTS_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "::error::jq is not installed. Please install it to continue."
    exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" != *"=="* ]]; then
        echo "::error::Invalid requirement format (must contain '=='): $line"
        exit 1
    fi
    package="${line%%==*}"
    version="${line##*==}"
    if [[ -z "$package" ]] || [[ -z "$version" ]]; then
        echo "::error::Invalid requirement format: $line"
        exit 1
    fi
    echo "Fetching wheel for $package==$version..."
    json_url="https://pypi.org/pypi/${package}/${version}/json"
    response=$(curl -sSL --fail --user-agent "Mozilla/5.0 (compatible; Debian-PPA-Build/1.0)" "$json_url")
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "::error::PyPI returned invalid JSON for $package==$version"
        exit 1
    fi
    source_url=$(echo "$response" | jq -r '.urls[] | select(.packagetype == "sdist") | .url' | head -n 1)
    if [[ -z "$source_url" ]]; then
        echo "::error::No compatible wheel found for $package==$version"
        exit 1
    fi
    filename=$(basename "$source_url")
    echo "Downloading: $filename"
    curl -sSL "$source_url" -o "${TMP_SRC_DIR}/vendor/sdist/${filename}"
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
# The tarball now correctly contains ONLY the upstream code + the vendored wheels.
tar -czf "$BUILD_DIR/$TARBALL_NAME" -C "$BUILD_DIR" "rayforge-${UPSTREAM_VERSION}"
echo "Created: $BUILD_DIR/$TARBALL_NAME"

# --- 4. Build the Package ---
# Now we add the debian directory and build.
cd "$BUILD_DIR"
cp -r "$ORIG_DIR/debian" "$TMP_SRC_DIR/"
cd "$TMP_SRC_DIR"

MAINTAINER_INFO=$(grep '^Maintainer:' debian/control | head -n 1 | sed 's/Maintainer: //')
export DEBEMAIL=$(echo "$MAINTAINER_INFO" | sed -E 's/.*<(.*)>.*/\1/')
export DEBFULLNAME=$(echo "$MAINTAINER_INFO" | sed -E 's/ <.*//')

# The `dch` commands are safe to run in the current environment
if [[ "${1:-}" == "--source" ]]; then
    TARGET_DISTRIBUTION="jammy"
    dch --newversion "${UPSTREAM_VERSION}-1~ppa1~${TARGET_DISTRIBUTION}1" --distribution "$TARGET_DISTRIBUTION" "New PPA release ${UPSTREAM_VERSION}."
    env -i \
        HOME="$HOME" \
        PATH="/usr/sbin:/usr/bin:/sbin:/bin" \
        DEBEMAIL="$DEBEMAIL" \
        DEBFULLNAME="$DEBFULLNAME" \
        SETUPTOOLS_GIT_VERSIONING_VERSION="$UPSTREAM_VERSION" \
        dpkg-buildpackage -S -us -uc
else
    dch --newversion "${UPSTREAM_VERSION}-1~local1" "New local build ${UPSTREAM_VERSION}."
    env -i \
        HOME="$HOME" \
        PATH="/usr/sbin:/usr/bin:/sbin:/bin" \
        DEBEMAIL="$DEBEMAIL" \
        DEBFULLNAME="$DEBFULLNAME" \
        SETUPTOOLS_GIT_VERSIONING_VERSION="$UPSTREAM_VERSION" \
        dpkg-buildpackage -b -us -uc
fi

# --- 5. Copy Artifacts ---
echo "--- Copying build artifacts back to project's dist/ directory ---"
mkdir -p "$ORIG_DIR/dist"
find "$BUILD_DIR" -maxdepth 1 -name 'rayforge*' -type f -exec cp -v {} "$ORIG_DIR/dist/" \;
