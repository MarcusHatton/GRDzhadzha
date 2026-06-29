#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-2}"
RUNNER="${RUNNER:-srun}"
DEFAULT_ENV="/home/mjhatto/Chombo/env_grchombo.sh"

if [[ "${GRHD_SOURCE_ENV:-1}" != "0" && -f "$DEFAULT_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$DEFAULT_ENV"
fi

export GRCHOMBO_SOURCE="${GRCHOMBO_SOURCE:-/home/mjhatto/GRChombo/Source}"

run_case() {
    local rel_dir="$1"
    local ebase="$2"
    local ranks="$3"
    local abs_dir="$ROOT_DIR/$rel_dir"

    echo "==> Building $rel_dir"
    GRCHOMBO_SOURCE="$GRCHOMBO_SOURCE" make -C "$abs_dir" all -j"$JOBS"

    local exe
    exe="$(find "$abs_dir" -maxdepth 1 -type f -name "${ebase}*.ex" | sort | tail -n 1)"
    if [[ -z "$exe" ]]; then
        echo "No executable matching ${ebase}*.ex in $rel_dir" >&2
        return 1
    fi

    echo "==> Running $rel_dir with $ranks MPI rank(s)"
    "$RUNNER" -n "$ranks" "$exe"
}

run_case "Tests/GRHDBinaryPNTest" "GRHDBinaryPNTest" 1
run_case "Tests/GRHDFixedBGTest" "GRHDFixedBGTest" 1
run_case "Tests/GRHDMichelAccretion" "GRHDMichelAccretion" 1
run_case "Tests/GRHDLevelDataMichelAccretion" "GRHDLevelDataMichelAccretion" 2
run_case "Tests/GRHDLevelDataBinaryPNSmoke" "GRHDLevelDataBinaryPNSmoke" 2
run_case "Tests/GRHDAMRHookCompileTest" "GRHDAMRHookCompileTest" 1
run_case "Tests/GRHDFixedBGAMRHookTest" "GRHDFixedBGAMRHookTest" 2
run_case "Examples/GRHDLevelDataKerrSchildDisk" "Main_GRHDLevelDataKerrSchildDisk" 2
run_case "Examples/GRHDLevelDataBinaryPNDisk" "Main_GRHDLevelDataBinaryPNDisk" 2

echo "GRHD fixed-background smoke suite passed."
