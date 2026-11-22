#!/bin/bash
# Ensures that the account owning GH_TOKEN has a fork of microsoft/winget-pkgs
# whose default branch matches upstream.
#
# "wingetcreate submit" syncs the fork via GitHub's merge-upstream API, but
# that API fails (422) when the fork has diverged from winget-pkgs, which
# moves too fast to keep up between releases
# (https://github.com/microsoft/winget-create/issues/710). Resetting the
# branch ref to the upstream HEAD works in all cases and is safe here because
# wingetcreate never commits to the fork's default branch.
set -e

UPSTREAM="repos/microsoft/winget-pkgs"
LOGIN="$(gh api user --jq .login)"
FORK="repos/$LOGIN/winget-pkgs"

if ! gh api "$FORK" >/dev/null 2>&1; then
    gh api --method POST "$UPSTREAM/forks" >/dev/null
fi

# Forks are created asynchronously, so the default branch may not be
# queryable right away.
BRANCH=""
for _ in $(seq 1 30); do
    BRANCH="$(gh api "$FORK" --jq .default_branch 2>/dev/null || true)"
    if [ -n "$BRANCH" ] && gh api "$FORK/git/ref/heads/$BRANCH" >/dev/null 2>&1; then
        break
    fi
    BRANCH=""
    sleep 10
done
if [ -z "$BRANCH" ]; then
    echo "Timed out waiting for the default branch of $LOGIN/winget-pkgs" >&2
    exit 1
fi

UPSTREAM_BRANCH="$(gh api "$UPSTREAM" --jq .default_branch)"
UPSTREAM_SHA="$(gh api "$UPSTREAM/git/ref/heads/$UPSTREAM_BRANCH" --jq .object.sha)"
FORK_SHA="$(gh api "$FORK/git/ref/heads/$BRANCH" --jq .object.sha)"

if [ "$UPSTREAM_SHA" = "$FORK_SHA" ]; then
    echo "Fork $LOGIN/winget-pkgs is already in sync with upstream ($BRANCH)"
    exit 0
fi

gh api --method PATCH "$FORK/git/refs/heads/$BRANCH" \
    -f sha="$UPSTREAM_SHA" -F force=true >/dev/null
echo "Reset $LOGIN/winget-pkgs/$BRANCH to upstream $UPSTREAM_SHA"
