#!/bin/bash
# Pooled cosine to the reference, jackknife SE and gradient noise scale of every arm recorded by
# scripts/gradprobe/math.sh (FAMILY=math: Tables 2, 18-20) or scripts/gradprobe/lcb.sh (FAMILY=lcb:
# Tables 3, 21) for one MODEL. Reads the dumps in NPARTS parallel processes, each over a disjoint set
# of tensors, and writes $OUT_ROOT/<model>/analysis/report.json.
#   FAMILY=math MODEL=Qwen/Qwen3-8B NPARTS=8 bash scripts/gradprobe/analyze.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

FAMILY=${FAMILY:-math}
MODEL=${MODEL:-Qwen/Qwen3-8B}
SEEDS=${SEEDS:-0 1 2 3}
NPARTS=${NPARTS:-8}
NAME=$(basename "$MODEL")
ROOT=${OUT_ROOT:-$REPO/outputs/gradprobe/$FAMILY}/$NAME
OUT=$ROOT/analysis

ARGS=(--seeds $SEEDS)
add() {  # <label> <glob>: only arms that were recorded
    if compgen -G "$2" >/dev/null; then ARGS+=(--method "$1=$2"); fi
}
case "$FAMILY" in
    math)
        case "$NAME" in Olmo*) REFERENCE_THINKING=True ;; *) REFERENCE_THINKING=False ;; esac
        ARGS+=(--reference "$ROOT/reference/grpo-model-$NAME-enable_thinking-$REFERENCE_THINKING-seed-*")
        add GRPO_think "$ROOT/train/grpo-model-$NAME-enable_thinking-True-seed-*"
        add GRPO_nothink "$ROOT/train/grpo-model-$NAME-enable_thinking-False-seed-*"
        add SFT "$ROOT/train/sft-model-$NAME-seed-*"
        # OPSD_<teacher thinking>_<z>: z is the reference solution, none, or another problem's (shuffled).
        for tt in True False; do
            for z in normal none swap; do
                label=OPSD_$([ $tt = True ] && echo think || echo nothink)_$(
                    case $z in normal) echo ref ;; none) echo none ;; swap) echo shuffled ;; esac)
                add "$label" "$ROOT/train/opsd-model-$NAME-teacher_thinking-$tt-privileged_mode-$z-seed-*"
            done
        done ;;
    lcb)
        ARGS+=(--reference "$ROOT/reference/lcb-grpo-$NAME-seed*")
        add GRPO "$ROOT/train/lcb-grpo-$NAME-seed*"
        add SDPO "$ROOT/train/lcb-sdpo-$NAME-seed*" ;;
    *) echo "FAMILY must be math or lcb" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
rm -f "$OUT"/gram_part*.npz
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for ((k = 0; k < NPARTS; k++)); do
    python -m gradprobe.analyze gram "${ARGS[@]}" --part $k --nparts "$NPARTS" --out "$OUT/gram_part$k.npz" \
        >"$OUT/gram_part$k.log" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python -m gradprobe.analyze report "$OUT"/gram_part*.npz --json "$OUT/report.json"
