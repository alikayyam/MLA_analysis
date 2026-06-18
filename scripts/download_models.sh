#!/usr/bin/env bash
# Download DeepSeek MLA models for re-running experiments.
# Usage:
#   ./scripts/download_models.sh           # download all three models
#   ./scripts/download_models.sh lite      # V2-Lite only  (~30 GB, Exp 1/2/3)
#   ./scripts/download_models.sh v2        # V2 only       (~440 GB, Exp 1)
#   ./scripts/download_models.sh v3        # V3 only       (~642 GB, Exp 1)
#
# Estimated download times at ~65 MB/s HDD write speed:
#   V2-Lite:  ~8 min
#   V2:       ~1.9 h
#   V3:       ~2.8 h
#
# Requirements: huggingface_hub CLI  (pip install huggingface_hub)
#               huggingface-cli login  (or set HF_TOKEN env var)

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

download() {
    local repo_id="$1"
    local local_dir="$2"
    echo "==> Downloading $repo_id -> $local_dir"
    huggingface-cli download "$repo_id" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False
    echo "==> Done: $local_dir"
}

TARGET="${1:-all}"

case "$TARGET" in
    lite|all)
        download "deepseek-ai/DeepSeek-V2-Lite" "$MODELS_DIR/deepseek-v2-lite"
        ;&  # fall through only when TARGET=all
    v2)
        [[ "$TARGET" == "all" ]] || true
        if [[ "$TARGET" == "v2" || "$TARGET" == "all" ]]; then
            download "deepseek-ai/DeepSeek-V2" "$MODELS_DIR/deepseek-v2"
        fi
        ;;
    v3)
        download "deepseek-ai/DeepSeek-V3" "$MODELS_DIR/deepseek-v3"
        ;;
esac

if [[ "$TARGET" == "all" ]]; then
    download "deepseek-ai/DeepSeek-V3" "$MODELS_DIR/deepseek-v3"
fi

echo ""
echo "All requested models downloaded to $MODELS_DIR"
echo "Run experiments from the repo root, e.g.:"
echo "  python scripts/run_exp1_weight_audit.py --model deepseek-v2-lite"
echo "  python scripts/run_exp2_activation.py   --model deepseek-v2-lite"
echo "  python scripts/run_exp3_clamping.py     --model deepseek-v2-lite"
