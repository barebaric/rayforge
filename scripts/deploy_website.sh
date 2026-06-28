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
if [ -z "$IS_PRERELEASE" ]; then
  echo "Error: IS_PRERELEASE environment variable is not set."
  exit 1
fi

# Rewrite the deployment branch history so only tagged release snapshots
# and the current HEAD survive. Every deploy replaces the whole site tree
# so intermediate untagged commits carry no unique content and only bloat
# the repository. Run this from inside the deploy repo checkout.
prune_untagged_history() {
  local current_head total kept sha

  current_head=$(git rev-parse HEAD)
  total=$(git rev-list --count HEAD)

  # Without at least one tag anchor there is no way to distinguish a
  # release snapshot from a rolling refresh — pruning would collapse the
  # entire pre-tag history to a single commit. Wait until a tagged
  # release has been deployed, then this activates.
  if ! git tag -l | grep -q .; then
    echo "No release tags yet; skipping prune (need an anchor)."
    return 0
  fi

  # Count commits that will survive: every tagged commit + the current
  # HEAD (the rolling "latest" docs, even if untagged).
  kept=0
  while read -r sha; do
    local keep=no
    [ "$sha" = "$current_head" ] && keep=yes
    if [ "$keep" = no ] && git tag -l --points-at "$sha" | grep -q .; then
      keep=yes
    fi
    [ "$keep" = yes ] && kept=$((kept + 1))
  done < <(git rev-list HEAD)

  if [ "$kept" -ge "$total" ]; then
    echo "No untagged history to prune (keeping ${kept}/${total} commits)."
    return 0
  fi

  echo "Pruning history: keeping ${kept}/${total} tagged/latest commits."

  # Drop every commit that is neither tagged nor the current HEAD.
  # filter-branch rewires parents of surviving commits, and
  # --tag-name-filter cat re-points tags at the rewritten commits.
  # Because each deploy is a full snapshot, skipped commits carry no
  # unique tree content.
  PRUNE_HEAD="$current_head" \
    git filter-branch -f --tag-name-filter cat --commit-filter '
      keep=no
      [ "$GIT_COMMIT" = "$PRUNE_HEAD" ] && keep=yes
      if [ "$keep" = no ] && git tag -l --points-at "$GIT_COMMIT" | grep -q .; then
        keep=yes
      fi
      if [ "$keep" = yes ]; then
        git commit-tree "$@"
      else
        skip_commit "$@"
      fi
    ' -- --all

  # filter-branch does not touch the working tree; resync it to the
  # rewritten HEAD (the tree is identical, but the sha has changed).
  git reset --hard HEAD

  # Drop filter-branch backup refs so the old, bloated history is not
  # retained locally (the remote is rewritten by the force-push below).
  git for-each-ref --format='%(refname)' refs/original/ |
    while read -r ref; do git update-ref -d "$ref"; done
}

# Use absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/website/build"
DEPLOY_DIR="${PROJECT_ROOT}/build/deploy_repo"
WEBSITE_SRC_DIR="${PROJECT_ROOT}/website"

echo "Starting website deployment for version: ${DEPLOY_VERSION}"
echo "Tagged release: ${IS_TAGGED_RELEASE}"
echo "Pre-release: ${IS_PRERELEASE}"
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

# Sync pre-generated raygeo API docs into the website tree
echo "Syncing raygeo API docs..."
python3 "${PROJECT_ROOT}/scripts/update_api_docs.py"

# Install dependencies and build the Docusaurus site
echo "Installing dependencies..."
cd "${WEBSITE_SRC_DIR}"
npm install

# Strip the 'v' prefix from version for download links (e.g., v1.0.2 -> 1.0.2)
RAYFORGE_VERSION="${DEPLOY_VERSION#v}"
echo "Building static site with version: ${RAYFORGE_VERSION}"
RAYFORGE_VERSION="${RAYFORGE_VERSION}" IS_PRERELEASE="${IS_PRERELEASE}" npm run build

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

if [ "${IS_TAGGED_RELEASE}" = "true" ]; then
  echo "Tagging deploy commit as ${DEPLOY_VERSION}"
  git tag -f "${DEPLOY_VERSION}"
fi

# Drop intermediate untagged commits from the branch history so the
# repository does not bloat over time. Only tagged release snapshots and
# the latest HEAD are kept. Once the first release tag exists, every
# branch push collapses old untagged refreshes into the new HEAD.
prune_untagged_history

git push --force origin "${DEPLOY_BRANCH}"
git push --force origin --tags
)

echo "✅ Deployment successful!"
