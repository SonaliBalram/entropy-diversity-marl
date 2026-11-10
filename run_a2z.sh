#!/usr/bin/env bash
# Train arm A2z (augmented critic, beta = 0) for all four seeds.

# Stop if an unset variable is used.
set -u

# Run from the project root.
cd "$(dirname "$0")"

# Seeds, episode count and arm name for this sweep.
SEEDS="42 128 456 789"
EPISODES=50000
ARM="beta_0_augcritic"

# Training options, matched to the other arms so results are comparable.
FLAGS="--beta 0 --augment-critic --episodes ${EPISODES} \
       --batch-size 256 --update-every 2 --checkpoint-every 2000"

# Pre-flight: refuse to start if the Python environment is not ready.
python3 - <<'PY' || exit 1
import sys
try:
    import torch, numpy
except ImportError as e:
    sys.exit(f"REFUSING TO START: {e}. Did you source .venv/bin/activate?")
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"numpy {numpy.__version__}")
PY

echo
echo "===================================================================="
echo " ARM A2z  --  beta = 0, augmented critic  (the capacity control)"
echo "===================================================================="
echo " host      : $(hostname)"
echo " started   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " seeds     : ${SEEDS}"
echo " episodes  : ${EPISODES}"
echo " flags     : ${FLAGS}"
echo " guard     : test_reductions.py 22/22 (run it again if the"
echo "             trainer or the critic has been touched since)"
echo "===================================================================="
echo

# Running summary and failure flag for the whole sweep.
SUMMARY=""
FAILED=0
SWEEP_START=$(date +%s)

# Train each seed in turn.
for S in ${SEEDS}; do
    OUT="results/${ARM}/seed_${S}"
    LOG="train_exp3z_beta0_seed${S}.log"

    # Skip a seed whose results already exist.
    if [ -f "${OUT}/episode_rewards.npy" ]; then
        echo "[skip] seed ${S}: ${OUT}/episode_rewards.npy already exists"
        SUMMARY="${SUMMARY}
  seed ${S}   SKIPPED (already complete)"
        continue
    fi

    echo "--------------------------------------------------------------------"
    echo "[run ] seed ${S}  ->  ${LOG}"
    echo "       $(date '+%H:%M:%S')"
    echo "--------------------------------------------------------------------"

    # Start the log file with the command and machine details.
    T0=$(date +%s)
    {
        echo "# command : python3 exp3/train_exp3.py ${FLAGS} --seed ${S}"
        echo "# started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# host    : $(hostname)"
        echo "#"
    } > "${LOG}"

    # Train this seed, appending all output to its log.
    python3 exp3/train_exp3.py ${FLAGS} --seed "${S}" >> "${LOG}" 2>&1
    RC=$?
    T1=$(date +%s)
    HOURS=$(python3 -c "print(f'{($T1 - $T0)/3600:.2f}')")

    # Report whether the seed finished successfully.
    if [ ${RC} -eq 0 ] && [ -f "${OUT}/episode_rewards.npy" ]; then
        echo "[done] seed ${S}  ${HOURS} h"
        SUMMARY="${SUMMARY}
  seed ${S}   OK        ${HOURS} h"
    else
        echo "[FAIL] seed ${S}  exit ${RC} after ${HOURS} h  -- see ${LOG}"
        echo "       last 20 lines:"
        tail -20 "${LOG}" | sed 's/^/       /'
        SUMMARY="${SUMMARY}
  seed ${S}   FAILED    exit ${RC}, ${HOURS} h"
        FAILED=1
    fi
    echo
done

# Print the summary and next steps.
TOTAL=$(python3 -c "print(f'{($(date +%s) - $SWEEP_START)/3600:.2f}')")

echo "===================================================================="
echo " A2z SWEEP COMPLETE"
echo "===================================================================="
echo "${SUMMARY}"
echo
echo "  total wall clock : ${TOTAL} h"
echo "  finished         : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  results          : results/${ARM}/"
echo "  checkpoints      : checkpoints/${ARM}/"
echo
if [ ${FAILED} -eq 0 ]; then
    echo "  NEXT: exp3_ladder.py over the new checkpoints (the cache means"
    echo "        only the new ones are evaluated), then exp3_outcome.py"
    echo "        against beta_3_augcritic to separate capacity from repair."
else
    echo "  ONE OR MORE SEEDS FAILED -- do not analyse a partial sweep."
fi
echo "===================================================================="

exit ${FAILED}
