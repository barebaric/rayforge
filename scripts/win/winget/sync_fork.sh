#!/bin/bash
# Ensures that the account owning GH_TOKEN has a fork of microsoft/winget-pkgs
# whose default branch matches upstream.
#
# "wingetcreate submit" syncs the fork via GitHub's merge-upstream API, but
# that API fails (422) when the fork has fallen too far behind winget-pkgs,
# which moves too fast to keep up between releases
# (https://github.com/microsoft/winget-create/issues/710). Resetting the
# branch ref to the upstream HEAD works in all cases and is safe here because
# wingetcreate never commits to the fork's default branch.
#
# GitHub refuses this ref update when the upstream commits touch
# .github/workflows/ and the token lacks the workflow scope, so
# check_token.sh must run first.
set -e

UPSTREAM="repos/microsoft/winget-pkgs"
LOGIN="$(gh api user --jq .login)"
FORK="repos/$LOGIN/winget-pkgs"

if ! gh api "$FORK" >/dev/null 2>&1; then
    echo "Creating fork $LOGIN/winget-pkgs"
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

echo "Resetting $LOGIN/winget-pkgs/$BRANCH to upstream $UPSTREAM_SHA"
if ! gh api --method PATCH "$FORK/git/refs/heads/$BRANCH" \
    -f sha="$UPSTREAM_SHA" -F force=true >/dev/null; then
    cat >&2 <<EOF
Failed to reset $LOGIN/winget-pkgs/$BRANCH to upstream $UPSTREAM_SHA.

A 404 here usually means WINGET_PAT lacks the workflow scope needed to
update files under .github/workflows/; see check_token.sh.
EOF
    exit 1
fi
echo "Fork $LOGIN/winget-pkgs/$BRANCH now matches upstream"
