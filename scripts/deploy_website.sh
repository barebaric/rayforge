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

# Use absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/website/build"
DEPLOY_DIR="${PROJECT_ROOT}/build/deploy_repo"
WEBSITE_SRC_DIR="${PROJECT_ROOT}/website"

echo "Starting website deployment for version: ${DEPLOY_VERSION}"
echo "Tagged release: ${IS_TAGGED_RELEASE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "Deploy directory: ${DEPLOY_DIR}"

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

# Strip the 'v' prefix from version for download links (e.g., v1.0.2 -> 1.0.2)
RAYFORGE_VERSION="${DEPLOY_VERSION#v}"
export RAYFORGE_VERSION
echo "Building static site with version: ${RAYFORGE_VERSION}"
npm run build

# Verify build output
echo "Build output:"
ls -la "${BUILD_DIR}/"

# Deploy: sync build output to deployment directory
# Exclude .git (repo data), .github (workflows), .well-known (domain verification)
echo "Deploying built site to ${DEPLOY_DIR}"
rsync -av --delete --exclude '.git' --exclude '.github' --exclude '.well-known' "${BUILD_DIR}/" "${DEPLOY_DIR}/"

# Verify deployment
echo "Deployed content:"
ls -la "${DEPLOY_DIR}/"

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
