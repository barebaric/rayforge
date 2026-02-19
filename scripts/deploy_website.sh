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
BUILD_DIR="website/build"
DEPLOY_DIR="build/deploy_repo"
WEBSITE_SRC_DIR="website"

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

# Install dependencies and build the Docusaurus site
echo "Installing dependencies..."
cd "${WEBSITE_SRC_DIR}"
npm install

echo "Building static site..."
npm run build
cd ..

# Post-build: Deploy the final site by merging files
# This rsync command adds the new build without deleting existing content
echo "Deploying built site to ${DEPLOY_DIR}"
rsync -a --delete "${BUILD_DIR}/" "${DEPLOY_DIR}/"

# Commit and Push to Deployment Repository
echo "Committing and pushing changes..."
(
  cd "${DEPLOY_DIR}"
  echo "Changed to folder $(pwd)"

  # Abort if this is not a git repository.
  if [ ! -d ".git" ]; then
    echo "CRITICAL ERROR: The deployment directory is not a Git repository. Aborting."
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
