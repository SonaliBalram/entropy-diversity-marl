#!/usr/bin/env bash
#
# run_a2z.sh -- arm A2z: beta = 0 with the augmented critic.
#
# THE CAPACITY CONTROL.
#
# A2c differs from A2 in two ways at once: the critic is wider,
# AND the terminal bonus became a deterministic function of the
# critic's own input. Every A2c effect -- the peak delayed to
# 26,731, the pessimism ratio falling 1.636 -> 1.094 -- is
# therefore confounded between "a wider critic changes learning"
# and "a wider critic makes THE BONUS learnable".
#
# This arm removes the bonus and keeps the width. If it
# reproduces A2c's delayed peak and calibration gain, those were
# pure capacity. If it looks like A0 instead, the augmentation
# mattered because it made the bonus learnable.
#
# Results -> results/beta_0_augcritic/seed_<S>/
# Logs    -> train_exp3z_beta0_seed<S>.log
#
# Usage:
#     source .venv/bin/activate
#     caffeinate -i ./run_a2z.sh          # -i stops idle sleep
#
# Safe to re-run: a seed whose results already exist is skipped,
# so an interrupted sweep resumes where it stopped.

set -u

cd "$(dirname "$0")"

SEEDS="42 128 456 789"
EPISODES=50000
ARM="beta_0_augcritic"

# The comparability contract. train_exp3.py DEFAULTS to
# --batch-size 1024 --update-every 1, but every arm this will be
# compared against used 256 and 2. Passing them explicitly is the
# difference between a control and a wasted six hours.
FLAGS="--beta 0 --augment-critic --episodes ${EPISODES} \
       --batch-size 256 --update-every 2 --checkpoint-every 2000"

# ----------------------------------------------------------------
# PRE-FLIGHT
# ----------------------------------------------------------------
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

SUMMARY=""
FAILED=0
SWEEP_START=$(date +%s)

for S in ${SEEDS}; do
    OUT="results/${ARM}/seed_${S}"
    LOG="train_exp3z_beta0_seed${S}.log"

    # Arrays are written only after training completes, so their
    # presence means the seed finished rather than merely started.
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

    T0=$(date +%s)
    {
        echo "# command : python3 exp3/train_exp3.py ${FLAGS} --seed ${S}"
        echo "# started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# host    : $(hostname)"
        echo "#"
    } > "${LOG}"

    # shellcheck disable=SC2086
    python3 exp3/train_exp3.py ${FLAGS} --seed "${S}" >> "${LOG}" 2>&1
    RC=$?
    T1=$(date +%s)
    HOURS=$(python3 -c "print(f'{($T1 - $T0)/3600:.2f}')")

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
