# Pre-launch checks for train_a2w.py, run in a temporary copy of the project.

import os
import sys
import tempfile
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from windowed_diversity import windowed_diversity

SKIP = {"results", "checkpoints", "logs", ".git", ".venv", "archive", ".reduction_test"}
COMMON = ["--seed", "42", "--print-every", "100000"]


# Build a temporary copy of the project from symlinks.
def make_mirror():
    m = tempfile.mkdtemp(prefix="a2w_test_")
    for name in os.listdir(ROOT):
        if name not in SKIP:
            os.symlink(os.path.join(ROOT, name), os.path.join(m, name))
    os.makedirs(os.path.join(m, "results"))
    os.makedirs(os.path.join(m, "checkpoints"))
    return m


# Run a short training job in the mirror.
def run(mirror, script, extra):
    cmd = [sys.executable, script] + COMMON + extra
    print("  running:", " ".join(cmd[1:]), flush=True)
    p = subprocess.run(cmd, cwd=mirror, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit(f"  FAILED TO RUN: {script}")


# Load one saved array from the mirror's results.
def load(mirror, exp, name):
    return np.load(os.path.join(mirror, "results", exp, "seed_42", name + ".npy"))


# Run every check and print PASS or FAIL.
def main():
    mirror = make_mirror()
    print(f"Mirror folder (throwaway): {mirror}\n")
    ok_all = True

    print("CHECK 1  --n-windows 1 vs old train_exp3.py (beta 3, 300 episodes)")
    run(mirror, "exp3/train_exp3.py",
        ["--beta", "3", "--episodes", "300", "--batch-size", "256", "--update-every", "2"])
    run(mirror, "exp3_windowed/train_a2w.py",
        ["--beta", "3", "--n-windows", "1", "--episodes", "300"])
    per_ep = ["episode_rewards", "episode_shaped_rewards", "episode_diversity_norm",
              "episode_discounted", "episode_value_avg", "running_mean_rewards"]
    per_up = ["critic_losses", "actor_losses", "mean_q_values", "policy_stds",
              "critic_disagreements", "update_steps"]
    worst_warm, worst_learn, exact = 0.0, 0.0, True
    for k in per_ep:
        a, b = load(mirror, "beta_3", k), load(mirror, "beta_3_w1", k)
        worst_warm = max(worst_warm, float(np.max(np.abs(a[:200] - b[:200]))))
        worst_learn = max(worst_learn, float(np.max(np.abs(a[200:] - b[200:]))))
        exact &= np.array_equal(a, b)
    for k in per_up:
        a, b = load(mirror, "beta_3", k), load(mirror, "beta_3_w1", k)
        if a.shape != b.shape:
            worst_learn = float("inf")
            continue
        worst_learn = max(worst_learn, float(np.max(np.abs(a - b))) if a.size else 0.0)
        exact &= np.array_equal(a, b)
    ok1 = worst_warm < 1e-12 and worst_learn < 1e-6
    ok_all &= ok1
    print(f"  warmup max |diff| = {worst_warm:.2e}   learning max |diff| = {worst_learn:.2e}"
          f"   {'(bit-identical)' if exact else ''}")
    print(f"  CHECK 1: {'PASS' if ok1 else 'FAIL'}\n")

    print("CHECK 2  --beta 0 vs train_baseline.py (250 episodes)")
    run(mirror, "train_baseline.py",
        ["--episodes", "250", "--batch-size", "256", "--update-every", "2"])
    run(mirror, "exp3_windowed/train_a2w.py", ["--beta", "0", "--episodes", "250"])
    first_diff, ok2 = [], True
    for k in ["episode_rewards", "episode_discounted", "episode_value_avg", "running_mean_rewards"]:
        a, b = load(mirror, "baseline", k), load(mirror, "beta_0_w5", k)
        ok2 &= np.array_equal(a[:200], b[:200])
        d = np.nonzero(a != b)[0]
        first_diff.append(int(d[0]) + 1 if d.size else None)
    ok_all &= ok2
    print(f"  identical on episodes 1-200: {ok2}   first differing episode: {first_diff} "
          f"(expected 201 or later)")
    print(f"  CHECK 2: {'PASS' if ok2 else 'FAIL'}\n")

    print("CHECK 3  bonus placement, A2 with 5 windows (250 episodes)")
    run(mirror, "exp3_windowed/train_a2w.py", ["--beta", "3", "--episodes", "250"])
    un = load(mirror, "beta_3_w5", "episode_rewards")
    sh = load(mirror, "beta_3_w5", "episode_shaped_rewards")
    dn = load(mirror, "beta_3_w5", "episode_diversity_norm")
    db = load(mirror, "beta_3_w5", "episode_diversity_bits")
    tok = load(mirror, "beta_3_w5", "tokens")
    e_bonus = float(np.max(np.abs(sh - un - 3.0 * dn)))
    e_tok = float(np.max(np.abs(np.array([windowed_diversity(t.T)[0] for t in tok]) - db)))
    ok3 = e_bonus < 1e-9 and e_tok < 1e-12
    ok_all &= ok3
    print(f"  max |shaped - unshaped - 3*D_norm| = {e_bonus:.2e}   "
          f"tokens reproduce D: {e_tok:.2e}   mean windowed D = {dn.mean():.3f}")
    print(f"  CHECK 3: {'PASS' if ok3 else 'FAIL'}\n")

    print("CHECK 4  A2c runs with the 11-number context (250 episodes)")
    run(mirror, "exp3_windowed/train_a2w.py",
        ["--beta", "3", "--episodes", "250", "--augment-critic"])
    cl = load(mirror, "beta_3_w5_augcritic", "critic_losses")
    ok4 = cl.size > 0 and bool(np.all(np.isfinite(cl)))
    ok_all &= ok4
    print(f"  assertion never fired (run completed); {cl.size} updates, "
          f"critic loss finite: {bool(np.all(np.isfinite(cl)))}, last {cl[-1]:.4f}")
    print(f"  CHECK 4: {'PASS' if ok4 else 'FAIL'}\n")

    print("=" * 60)
    print("ALL CHECKS PASS -- ready to launch" if ok_all else
          "AT LEAST ONE CHECK FAILED -- do not launch; send the output")
    print("=" * 60)


if __name__ == "__main__":
    main()
