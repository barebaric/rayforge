#!/bin/bash
set -e

# Configuration
if [ -z "$DEPLOY_VERSION" ]; then
  echo "Error: DEPLOY_VERSION environment variable is not set."
  exit 1
fi
if [ -z "$DEPLOY_REPO_URL" ]; then
  echo "Error: DEPLOY_REPO_URL environment variable is not set."
  exit 1
fi
if [ -z "$DEPLOY_BRANCH" ]; then
  echo "Error: DEPLOY_BRANCH environment variable is not set."
  exit 1
fi
if [ -z "$IS_TAGGED_RELEASE" ]; then
  echo "Error: IS_TAGGED_RELEASE environment variable is not set."
  exit 1
fi

# Paths
BUILD_DIR="build/website"
DEPLOY_DIR="build/deploy_repo"
TMP_SRC_DIR="build/tmp_source"
WEBSITE_SRC_DIR="website"
VERSIONS_FILE="${DEPLOY_DIR}/versions.json"
MKDOCS_CONFIG_BASE="mkdocs.yml"
MKDOCS_CONFIG_DEPLOY="mkdocs_deploy.yml"

echo "Starting website deployment for version: ${DEPLOY_VERSION}"
echo "Tagged release: ${IS_TAGGED_RELEASE}"

# Clone Deployment Repository
echo "Cloning deployment repository from ${DEPLOY_REPO_URL}..."
rm -rf "${DEPLOY_DIR}"
git clone "${DEPLOY_REPO_URL}" "${DEPLOY_DIR}"
git -C "${DEPLOY_DIR}" checkout -B "${DEPLOY_BRANCH}"

BOT_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
git -C "${DEPLOY_DIR}" config user.name "github-actions[bot]"
git -C "${DEPLOY_DIR}" config user.email "${BOT_EMAIL}"

# Pre-build: Prepare the source directory and generate config
python scripts/prepare_site_build.py \
  "${WEBSITE_SRC_DIR}" \
  "${TMP_SRC_DIR}" \
  "${MKDOCS_CONFIG_BASE}" \
  "${VERSIONS_FILE}" \
  "${DEPLOY_VERSION}" \
  "${MKDOCS_CONFIG_DEPLOY}"

# Build Site from the prepared source directory
echo "Building static site from prepared source..."
(
  cd "${TMP_SRC_DIR}"
  mkdocs build -f "${MKDOCS_CONFIG_DEPLOY}" --strict --clean \
    --site-dir "../website"
)
echo "Cleaning up temporary source directory..."
rm -rf "${TMP_SRC_DIR}"

# Post-build: Deploy the final site by merging files
# This rsync command adds the new version and updates root files (e.g.
# index.html, assets) without deleting existing version directories.
echo "Deploying built site to ${DEPLOY_DIR}"
rsync -a "${BUILD_DIR}/" "${DEPLOY_DIR}/"

# Update versions.json for tagged releases
if [ "${IS_TAGGED_RELEASE}" == "true" ]; then
  echo "This is a tagged release. Updating versions.json..."
  python scripts/update_site_versions.py \
    "${DEPLOY_VERSION}" "${VERSIONS_FILE}"
fi

# Commit and Push to Deployment Repository
echo "Committing and pushing changes..."
(
  cd "${DEPLOY_DIR}"

  # Abort if this is not a git repository.
  if [ ! -d ".git" ]; then
    echo "CRITICAL ERROR: The deployment directory is not a Git" \
         "repository. Aborting."
    exit 1
  fi
  if [ ! -d ".github" ]; then
    echo "CRITICAL ERROR: The deployment directory would delete" \
         ".github. Aborting."
    exit 1
  fi

  # Using --all to stage deletions as well
  git add --all .
  if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to deploy. Exiting."
    exit 0
  fi

  git commit -m "Deploy website content for ${DEPLOY_VERSION}"
  git push origin "${DEPLOY_BRANCH}"
)

echo "✅ Deployment successful!"
