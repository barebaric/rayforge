#!/bin/bash
# Renders the winget manifests for a release into a directory that can be
# submitted to microsoft/winget-pkgs using "wingetcreate submit".
set -e

# The application version string, the installer download URL, the installer
# SHA256 checksum, and the output directory for the rendered manifests.
VERSION="$1"
INSTALLER_URL="$2"
INSTALLER_SHA256="$3"
OUTDIR="$4"

if [ -z "$VERSION" ] || [ -z "$INSTALLER_URL" ] || [ -z "$INSTALLER_SHA256" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: $0 <version> <installer-url> <installer-sha256> <outdir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DATE="$(date +%Y-%m-%d)"
RELEASE_NOTES_URL="https://github.com/barebaric/rayforge/releases/tag/${VERSION}"

# Versions like "1.13.0-beta1" must be published as winget prereleases,
# otherwise winget would offer them over the matching stable release.
if [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+- ]]; then
    PRERELEASE="Prerelease: true"
else
    PRERELEASE=""
fi

mkdir -p "$OUTDIR"

render() {
    sed -e "s|@VERSION@|$VERSION|g" \
        -e "s|@INSTALLER_URL@|$INSTALLER_URL|g" \
        -e "s|@INSTALLER_SHA256@|$INSTALLER_SHA256|g" \
        -e "s|@RELEASE_DATE@|$RELEASE_DATE|g" \
        -e "s|@RELEASE_NOTES_URL@|$RELEASE_NOTES_URL|g" \
        -e "s|@PRERELEASE@|$PRERELEASE|g" \
        "$1" | sed '/^[[:space:]]*$/d' > "$OUTDIR/$2"
}

render "$SCRIPT_DIR/Rayforge.Rayforge.yaml.in" "Rayforge.Rayforge.yaml"
render "$SCRIPT_DIR/Rayforge.Rayforge.installer.yaml.in" "Rayforge.Rayforge.installer.yaml"

# Renders every locale manifest, including the en-US default locale.
for template in "$SCRIPT_DIR"/Rayforge.Rayforge.locale.*.yaml.in; do
    filename="$(basename "$template" .in)"
    render "$template" "$filename"
done

echo "Rendered winget manifests for version $VERSION into $OUTDIR"
