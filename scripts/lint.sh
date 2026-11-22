#!/usr/bin/env bash
# Runs the project's linters. Without arguments all checks run in
# parallel; with an argument only the named check runs.
set -u

cd "$(dirname "$0")/.." || exit 1

run_ruff() {
    ruff check rayforge tests scripts "$@"
}

run_flake() {
    flake8 --ignore=E127,E128,E121,E123,E126,E203,E226,E24,E704,W503,W504 \
        --builtins=_ rayforge tests "$@"
}

run_pyflakes() {
    PYFLAKES_BUILTINS=_ pyflakes rayforge tests "$@"
}

run_pyright() {
    local threads
    threads=$(python -c 'import os; print(min(os.cpu_count() or 1, 8))')
    VIRTUAL_ENV=.pixi/envs/default pyright \
        --pythonpath .pixi/envs/default/bin/python \
        --threads "$threads" "$@"
}

run_validate_metainfo() {
    appstreamcli validate --no-net \
        data/org.rayforge.rayforge.metainfo.xml "$@"
}

run_check() {
    local check=$1
    shift
    case "$check" in
        ruff) run_ruff "$@" ;;
        flake) run_flake "$@" ;;
        pyflakes) run_pyflakes "$@" ;;
        pyright) run_pyright "$@" ;;
        validate-metainfo) run_validate_metainfo "$@" ;;
        *)
            echo "Unknown lint check: $check" >&2
            return 2
            ;;
    esac
}

if [ "$#" -gt 0 ]; then
    run_check "$@"
    exit
fi

logdir=$(mktemp -d) || exit 1
trap 'rm -rf "$logdir"' EXIT

checks=(ruff flake pyflakes pyright validate-metainfo)
pids=()

for check in "${checks[@]}"; do
    run_check "$check" >"$logdir/$check.log" 2>&1 &
    pids+=("$!")
done

status=0
for i in "${!checks[@]}"; do
    if ! wait "${pids[$i]}"; then
        status=1
    fi
done

for check in "${checks[@]}"; do
    echo "--- $check ---"
    cat "$logdir/$check.log"
done

exit "$status"
