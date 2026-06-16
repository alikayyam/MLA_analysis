#!/usr/bin/env bash
# Waits for Exp 2 (PID 1927892) to finish, then launches Exp 3.
EXP2_PID=1953506

echo "[$(date)] Waiting for Exp 2 (PID $EXP2_PID) to finish..."
while kill -0 $EXP2_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Exp 2 done. Launching Exp 3..."
cd /home/akayyam/MLA
uv run scripts/run_exp3_clamping.py \
    --model deepseek-v2-lite \
    --model-dir models/deepseek-v2-lite \
    --exp1-results results/deepseek-v2-lite/exp1_metrics \
    --output-dir results \
    --schedules middle_block global progressive_outward \
    --skip-downstream \
    --max-ppl-samples 50 \
    > logs/exp3_v2lite.log 2>&1

echo "[$(date)] Exp 3 finished. See logs/exp3_v2lite.log"
