#!/bin/bash
# Verifies that WINGET_PAT can sync the winget-pkgs fork and submit a PR.
#
# GitHub refuses any write that adds or changes files under
# .github/workflows/ unless the token carries the workflow scope, including
# syncing a fork whose upstream commits touch a workflow file (for example
# .github/workflows/scriptAnalyzer.yaml in microsoft/winget-pkgs).
# wingetcreate reports that refusal as a generic fork sync failure, so
# reject an under-scoped token before doing any work.
set -e

TOKEN_URL="https://github.com/settings/tokens"

if [ -z "${GH_TOKEN:-}" ]; then
    echo "GH_TOKEN is empty; set it to a classic personal access token" >&2
    exit 1
fi

if ! headers="$(gh api -i user 2>&1)"; then
    echo "Could not query the WINGET_PAT owner: $headers" >&2
    exit 1
fi

scopes="$(printf '%s\n' "$headers" |
    tr -d '\r' |
    sed -n 's/^[Xx]-[Oo][Aa]uth-[Ss]copes:[[:space:]]*//p' |
    tr -d ' ')"

if [ -z "$scopes" ]; then
    cat >&2 <<EOF
WINGET_PAT does not report OAuth scopes, so it is not a classic personal
access token. wingetcreate needs a classic token with the public_repo and
workflow scopes. Create one at $TOKEN_URL and update the WINGET_PAT
repository secret.
EOF
    exit 1
fi

missing=""
has_scope() {
    case ",$scopes," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# The full repo scope is a superset of public_repo.
if ! has_scope public_repo && ! has_scope repo; then
    missing="$missing public_repo"
fi
if ! has_scope workflow; then
    missing="$missing workflow"
fi

if [ -n "$missing" ]; then
    cat >&2 <<EOF
WINGET_PAT is missing the following scopes:$missing

Create a classic personal access token with the public_repo and workflow
scopes at $TOKEN_URL and update the WINGET_PAT repository secret.

The workflow scope is required because syncing the winget-pkgs fork
replays upstream commits that modify .github/workflows/.
EOF
    exit 1
fi

echo "WINGET_PAT has the required scopes"
