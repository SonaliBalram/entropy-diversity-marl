#!/usr/bin/env bash
#
# run_a1r.sh -- arm A1r: ownership-gated entropy allocation.
#
# Same entropy budget as A1 (alpha = 0.01), redistributed across states
# by a frozen gate built on two Voronoi diagrams:
#
#     alpha_eff,i = alpha * w_i / Z      w_i = 1 / (1 + exp(-G_i / kappa))
#     G_i = max( g_i , -m_i )
#
# kappa and Z are read from results/a1r_pilot/a1r_constants.json and are
# NEVER re-estimated. The pre-flight below refuses to start if that file
# is missing, because choosing kappa after seeing training is exactly the
# researcher degree of freedom the pre-registration exists to remove.
#
# Results -> results/a1r_both/seed_<S>/
# Logs    -> A1r/a1r_both_seed<S>.log
#
# Usage:
#     source .venv/bin/activate
#     caffeinate -i ./A1r/run_a1r.sh      # -i stops idle sleep
#
# Safe to re-run: a seed whose results already exist is skipped, so an
# interrupted sweep resumes where it stopped.

set -u

# the script lives in A1r/, the project root is one level up
cd "$(dirname "$0")/.."

SEEDS="42 128 456 789"
EPISODES=50000
GATE="both"
ALPHA=0.01
ARM="a1r_${GATE}"

# The comparability contract. Held at Experiment 2's settings, with
# alpha_eff(o) as the single changed factor. train_a1r.py inherits
# exp2's defaults of --batch-size 1024 --update-every 1, so passing 256
# and 2 explicitly is the difference between a comparable arm and a
# wasted overnight run.
FLAGS="--alpha ${ALPHA} --gate ${GATE} --episodes ${EPISODES} \
       --batch-size 256 --update-every 2 --checkpoint-every 2000"

# ----------------------------------------------------------------
# PRE-FLIGHT
# ----------------------------------------------------------------
python3 - <<'PY' || exit 1
import sys, os, json
try:
    import torch, numpy
except ImportError as e:
    sys.exit(f"REFUSING TO START: {e}. Did you source .venv/bin/activate?")
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"numpy {numpy.__version__}")

p = "results/a1r_pilot/a1r_constants.json"
if not os.path.exists(p):
    sys.exit("REFUSING TO START: no " + p + "\n"
             "  Run  python A1r/a1r_pilot.py  first. It freezes kappa and\n"
             "  log Z from a reference set of states and reports the\n"
             "  kill-switch verdict.")
c = json.load(open(p))
if c.get("verdict", "").startswith("DO NOT RUN"):
    sys.exit(f"REFUSING TO START: pilot verdict is {c['verdict']!r}.\n"
             "  Re-read results/a1r_pilot/report.txt before overriding.")
print(f"gate      kappa {c['kappa']:.6f}   log Z {c['log_z']:.6f}   "
      f"mode {c['mode']}")
print(f"pilot     {c['reference']['n_samples']} reference states, "
      f"verdict {c['verdict']!r}")
print(f"          fitted {c['created']}")
PY

echo
echo "===================================================================="
echo " ARM A1r  --  ownership-gated entropy allocation"
echo "===================================================================="
echo " host      : $(hostname)"
echo " started   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " seeds     : ${SEEDS}"
echo " episodes  : ${EPISODES}"
echo " flags     : ${FLAGS}"
echo " guard     : test_reductions_a1r.py R1-R5 (run it again if"
echo "             train_a1r.py or maddpg_system_a1r.py has been touched)"
echo "===================================================================="
echo

SUMMARY=""
FAILED=0
SWEEP_START=$(date +%s)

for S in ${SEEDS}; do
    OUT="results/${ARM}/seed_${S}"
    LOG="A1r/${ARM}_seed${S}.log"

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
        echo "# command : python3 A1r/train_a1r.py ${FLAGS} --seed ${S}"
        echo "# started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# host    : $(hostname)"
        echo "#"
    } > "${LOG}"

    # shellcheck disable=SC2086
    python3 A1r/train_a1r.py ${FLAGS} --seed "${S}" >> "${LOG}" 2>&1
    RC=$?
    T1=$(date +%s)
    HOURS=$(python3 -c "print(f'{($T1 - $T0)/3600:.2f}')")

    if [ ${RC} -eq 0 ] && [ -f "${OUT}/episode_rewards.npy" ]; then
        # the two things that can silently invalidate the arm
        BUDGET=$(python3 -c "
import numpy as np
g = np.load('${OUT}/gate_geomeans.npy')
s = np.load('${OUT}/gate_spreads.npy')
print(f'budget {np.exp(np.log(g).mean())/${ALPHA}:.3f}x alpha, '
      f'spread {s.mean():.3f} dec (1.0 on the reference set)')
" 2>/dev/null || echo "gate logs unreadable")
        echo "[done] seed ${S}  ${HOURS} h   ${BUDGET}"
        SUMMARY="${SUMMARY}
  seed ${S}   OK        ${HOURS} h   ${BUDGET}"
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
echo " A1r SWEEP COMPLETE"
echo "===================================================================="
echo "${SUMMARY}"
echo
echo "  total wall clock : ${TOTAL} h"
echo "  finished         : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  results          : results/${ARM}/"
echo "  checkpoints      : checkpoints/${ARM}/"
echo
if [ ${FAILED} -eq 0 ]; then
    echo "  NEXT, in this order:"
    echo "    1. python A1r/a1r_tau.py --ckpt-root checkpoints/${ARM}"
    echo "       tau ~ 0 and tau ~ 1 BOTH predict no effect, so read tau"
    echo "       before looking at Delta D or you cannot interpret it."
    echo "    2. exp3_ladder.py then exp3_outcome.py against alpha_0.01,"
    echo "       matched-return Delta D. A1r adds no new analysis code."
    echo
    echo "  Check the budget column above. It is the realised geometric"
    echo "  mean of alpha_eff; Z is frozen, so drift away from 1.00x is"
    echo "  expected and must be REPORTED, not corrected. If the spread"
    echo "  column collapsed towards 0, the allocation flattened and A1r"
    echo "  became A1 -- say so rather than reporting a null."
else
    echo "  ONE OR MORE SEEDS FAILED -- do not analyse a partial sweep."
fi
echo "===================================================================="

exit ${FAILED}
