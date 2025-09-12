#!/bin/bash
set -ex

# --- Setup & Cleanup ---
BUILD_DIR=$(mktemp -d)
ORIG_DIR=$(pwd)
# The trap command ensures the temporary build directory is always removed on exit.
cleanup() {
    echo "--- Cleaning up temporary build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
    echo "Cleanup complete."
}
trap cleanup EXIT

# === All Git operations happen HERE, in the original repository ===

# --- 1. Dynamic Version Detection ---
echo "--- Determining version from Git repository ---"
if [[ -n "$GITHUB_REF_NAME" && "$GITHUB_REF_TYPE" == "tag" ]]; then
  UPSTREAM_VERSION_RAW="$GITHUB_REF_NAME"
else
# Create a compliant pre-release version string like 0.20.2~dev48~f12659d
  UPSTREAM_VERSION_RAW=$(git describe --tags --always --long | sed -e 's/^v//' -e 's/\([^-]*\)-\([0-9]*\)-g\([0-9a-f]*\)/\1~dev\2~\3/')
fi
UPSTREAM_VERSION="${UPSTREAM_VERSION_RAW#v}"
echo "Detected upstream version: ${UPSTREAM_VERSION}"

# --- 2. Create the Upstream Tarball (.orig.tar.gz) ---
# This is required for the 3.0 (quilt) source format.
TARBALL_NAME="rayforge_${UPSTREAM_VERSION}.orig.tar.gz"
# Use git archive to create a pristine tarball containing ONLY tracked files.
git archive --format=tar --prefix="rayforge-${UPSTREAM_VERSION}/" HEAD | gzip > "${BUILD_DIR}/${TARBALL_NAME}"
echo "Created pristine upstream tarball: ${BUILD_DIR}/${TARBALL_NAME}"

# === Build preparation happens HERE, in the temporary directory ===

# --- 3. Prepare the Clean Build Environment ---
echo "--- Preparing clean build directory ---"
# Unpack the pristine source code into the build directory
tar -xzf "${BUILD_DIR}/${TARBALL_NAME}" -C "$BUILD_DIR"
# Copy the Debian packaging files into the unpacked source tree
cp -a debian "${BUILD_DIR}/rayforge-${UPSTREAM_VERSION}/"

# --- 4. Run the Build Inside the Clean Directory ---
cd "${BUILD_DIR}/rayforge-${UPSTREAM_VERSION}"

# Prepare Environment for dch
MAINTAINER_INFO=$(grep '^Maintainer:' debian/control | head -n 1 | sed 's/Maintainer: //')
export DEBEMAIL=$(echo "$MAINTAINER_INFO" | sed 's/.*<\(.*\)>.*/\1/')
export DEBFULLNAME=$(echo "$MAINTAINER_INFO" | sed 's/ <.*//')

# Update Changelog and build the package
if [ "${1:-}" == "--source" ]; then
  # For PPA: Build an unsigned source package
  dch --newversion "${UPSTREAM_VERSION}-1~ppa1" "New PPA release ${UPSTREAM_VERSION}."
  debuild -S -us -uc
else
  # For Local: Build a binary package
  dch --newversion "${UPSTREAM_VERSION}-1~local1" "New local build ${UPSTREAM_VERSION}."
  debuild -b -us -uc
fi

# --- 5. Copy Artifacts Back to Original Project Directory ---
echo "--- Copying build artifacts back to project's dist/ directory ---"
mkdir -p "$ORIG_DIR/dist"
# debuild places artifacts in the parent of the build tree ($BUILD_DIR)
# Copy all generated package files to dist/
cp -v "${BUILD_DIR}"/rayforge_* "$ORIG_DIR/dist/"
