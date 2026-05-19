#!/usr/bin/env bash
#
# Judge-like Tier 1 validation for a macro placement submission.
#
# Differences from run_eval.sh:
#   - runs each IBM benchmark independently,
#   - enforces a per-benchmark timeout,
#   - defaults to the published judge CPU/memory envelope,
#   - keeps runtime air-gapped with --network none.
#
# Usage:
#   ./eval_docker/run_judge_like.sh <team_name> <placer_path> [extra_mount...]

set -euo pipefail

TEAM="${1:?Usage: $0 <team_name> <placer_path> [extra_mount...]}"
PLACER_PATH="${2:?Usage: $0 <team_name> <placer_path> [extra_mount...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${EVAL_IMAGE_NAME:-macro-place-eval}"
RESULTS_DIR="$REPO_ROOT/eval_docker/results/judge_like"
RUN_DIR="$RESULTS_DIR/$TEAM"

CPUS="${EVAL_CPUS:-16}"
MEMORY="${EVAL_MEMORY:-100g}"
TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-3600}"
GPUS="${EVAL_GPUS:-none}"
PASSTHROUGH_ENV_VARS="${EVAL_ENV_VARS:-}"

BENCHMARKS=(
    ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10
    ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18
)

mkdir -p "$RUN_DIR"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "=== Building evaluation Docker image ==="
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
fi

ABS_PLACER="$(cd "$REPO_ROOT" && realpath "$PLACER_PATH")"
PLACER_DIR="$(dirname "$ABS_PLACER")"
PLACER_FILE="$(basename "$ABS_PLACER")"

MOUNT_ARGS=("-v" "$PLACER_DIR:/submission:ro")
for extra in "$@"; do
    ABS_EXTRA="$(cd "$REPO_ROOT" && realpath "$extra")"
    BASE_EXTRA="$(basename "$ABS_EXTRA")"
    MOUNT_ARGS+=("-v" "$ABS_EXTRA:/submission/$BASE_EXTRA:ro")
done

GPU_ARGS=()
if [[ "$GPUS" != "none" && "$GPUS" != "0" ]]; then
    GPU_ARGS=("--gpus" "$GPUS")
fi

ENV_ARGS=()
if [[ -n "$PASSTHROUGH_ENV_VARS" ]]; then
    IFS=',' read -ra ENV_NAMES <<< "$PASSTHROUGH_ENV_VARS"
    for env_name in "${ENV_NAMES[@]}"; do
        env_name="$(xargs <<< "$env_name")"
        if [[ -n "$env_name" && -n "${!env_name-}" ]]; then
            ENV_ARGS+=("-e" "$env_name=${!env_name}")
        fi
    done
fi

SUMMARY="$RUN_DIR/summary.tsv"
COMBINED_LOG="$RUN_DIR/combined.log"
: > "$COMBINED_LOG"
printf "benchmark\tstatus\tproxy\toverlaps\truntime_seconds\n" > "$SUMMARY"

echo "=== Judge-like Tier 1 evaluation: $TEAM ===" | tee -a "$COMBINED_LOG"
echo "    Placer:         $PLACER_PATH" | tee -a "$COMBINED_LOG"
echo "    Image:          $IMAGE_NAME" | tee -a "$COMBINED_LOG"
echo "    Network:        none" | tee -a "$COMBINED_LOG"
echo "    CPUs:           $CPUS" | tee -a "$COMBINED_LOG"
echo "    Memory:         $MEMORY" | tee -a "$COMBINED_LOG"
echo "    Timeout/bench:  ${TIMEOUT_SECONDS}s" | tee -a "$COMBINED_LOG"
echo "    GPUs:           $GPUS" | tee -a "$COMBINED_LOG"
echo "" | tee -a "$COMBINED_LOG"

for bench in "${BENCHMARKS[@]}"; do
    bench_log="$RUN_DIR/$bench.log"
    echo "=== $bench ===" | tee -a "$COMBINED_LOG"

    set +e
    timeout "$TIMEOUT_SECONDS" docker run --rm \
        --network none \
        "${GPU_ARGS[@]}" \
        --memory "$MEMORY" \
        --cpus "$CPUS" \
        "${ENV_ARGS[@]}" \
        "${MOUNT_ARGS[@]}" \
        "$IMAGE_NAME" \
        "/submission/$PLACER_FILE" --benchmark "$bench" \
        2>&1 | tee "$bench_log"
    status="${PIPESTATUS[0]}"
    set -e

    cat "$bench_log" >> "$COMBINED_LOG"

    if [[ "$status" -eq 124 ]]; then
        printf "%s\ttimeout\t\t\t\n" "$bench" | tee -a "$SUMMARY"
        continue
    fi
    if [[ "$status" -ne 0 ]]; then
        printf "%s\terror\t\t\t\n" "$bench" | tee -a "$SUMMARY"
        continue
    fi

    result_line="$(grep -E "proxy=.*\\[[0-9.]+s\\]" "$bench_log" | tail -1 || true)"
    proxy="$(sed -nE 's/.*proxy=([0-9.]+).*/\1/p' <<< "$result_line")"
    runtime="$(sed -nE 's/.*\[([0-9.]+)s\].*/\1/p' <<< "$result_line")"
    overlaps="0"
    if grep -q "INVALID" <<< "$result_line"; then
        overlaps="$(sed -nE 's/.*INVALID \(([0-9]+) overlaps\).*/\1/p' <<< "$result_line")"
    fi
    printf "%s\tok\t%s\t%s\t%s\n" "$bench" "$proxy" "$overlaps" "$runtime" | tee -a "$SUMMARY"
done

awk -F '\t' '
    NR > 1 && $2 == "ok" {
        n += 1
        proxy += $3
        overlaps += $4
        runtime += $5
    }
    NR > 1 && $2 != "ok" {
        bad += 1
    }
    END {
        print ""
        print "=== Aggregate ==="
        if (n > 0) {
            printf("completed=%d failed=%d avg_proxy=%.4f overlaps=%d runtime=%.2fs\n", n, bad, proxy / n, overlaps, runtime)
        } else {
            printf("completed=0 failed=%d avg_proxy=NA overlaps=NA runtime=NA\n", bad)
        }
    }
' "$SUMMARY" | tee -a "$COMBINED_LOG"

echo ""
echo "=== Results saved under: $RUN_DIR ==="
