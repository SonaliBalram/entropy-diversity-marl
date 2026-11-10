import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
WARMUP_EPISODES = 150
CROSS_EPISODES = 260
COMMON = ["--seed", str(SEED), "--batch-size", "256",
          "--update-every", "2", "--print-every", "100000"]

WORK = os.path.join(HERE, ".reduction_test")


def build_sandbox():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(os.path.join(WORK, "exp2"))
    os.makedirs(os.path.join(WORK, "exp3"))

    for name in os.listdir(HERE):
        if name.endswith(".py") or name == "models":
            src = os.path.join(HERE, name)
            if os.path.isfile(src) or name == "models":
                os.symlink(src, os.path.join(WORK, name))
    for sub, name in [("exp2", "train_exp2.py"), ("exp3", "train_exp3.py")]:
        os.symlink(os.path.join(HERE, sub, name),
                   os.path.join(WORK, sub, name))


def train(script, extra, episodes):
    cmd = [sys.executable, os.path.join(WORK, script)] + extra + \
          COMMON + ["--episodes", str(episodes)]
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def load(experiment):
    d = os.path.join(WORK, "results", experiment, f"seed_{SEED}")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
        elif f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            for k in z.files:
                out[f"{f[:-4]}::{k}"] = z[k]
    return out


def compare(a, b, label):
    A, B = load(a), load(b)
    shared = sorted(set(A) & set(B))
    bad = []
    for k in shared:
        x, y = A[k], B[k]
        if x.shape != y.shape or not np.array_equal(x, y):
            bad.append(k)
    status = "PASS" if not bad else "FAIL"
    print(f"  [{status}] {label}: {len(shared) - len(bad)}/{len(shared)} "
          f"arrays bit-identical")
    for k in bad:
        print(f"          differs: {k}")
    return not bad


def main():
    build_sandbox()

    print("=" * 68)
    print("TEST 1  exact reduction inside the warmup window "
          f"({WARMUP_EPISODES} episodes, no gradient updates)")
    print("=" * 68)
    train("train_baseline.py", [], WARMUP_EPISODES)
    train(os.path.join("exp2", "train_exp2.py"),
          ["--alpha", "0.01"], WARMUP_EPISODES)
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0"], WARMUP_EPISODES)
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0.01"], WARMUP_EPISODES)

    ok = compare("baseline", "beta_0",
                 "--beta 0 --alpha 0    vs Experiment 1")
    ok &= compare("alpha_0.01", "beta_0_alpha_0.01",
                  "--beta 0 --alpha 0.01 vs Experiment 2")

    print()
    print("=" * 68)
    print("TEST 2  divergence after warmup begins at the first "
          "learning episode, and only there")
    print("=" * 68)
    shutil.rmtree(os.path.join(WORK, "results"))
    train("train_baseline.py", [], CROSS_EPISODES)
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0"], CROSS_EPISODES)
    x = load("baseline")["episode_rewards"]
    y = load("beta_0")["episode_rewards"]
    agree = int(np.argmin(x == y)) if not (x == y).all() else len(x)
    expected = 200
    good = agree == expected
    print(f"  [{'PASS' if good else 'FAIL'}] identical for the first "
          f"{agree} episodes; first difference at episode {agree + 1} "
          f"(expected {expected + 1})")
    if not good:
        print("          A DIFFERENT boundary means something other than the "
              "deferred buffer write is changing the run. Investigate before "
              "launching an arm.")
    ok &= good

    print()
    print("=" * 68)
    print("TEST 3  the bonus path, isolated inside the warmup window")
    print("=" * 68)
    shutil.rmtree(os.path.join(WORK, "results"))
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0", "--null-every", "50"],
          WARMUP_EPISODES)
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "5", "--alpha", "0", "--null-every", "50"],
          WARMUP_EPISODES)
    z0, z5 = load("beta_0"), load("beta_5")

    checks = [
        ("bonus does not alter the policy during warmup",
         np.array_equal(z0["episode_rewards"], z5["episode_rewards"])),
        ("D is identical between beta=0 and beta=5",
         np.array_equal(z0["episode_diversity_norm"],
                        z5["episode_diversity_norm"])),
        ("shaped == unshaped + beta * D_norm, exactly",
         np.array_equal(z5["episode_shaped_rewards"],
                        z5["episode_rewards"]
                        + 5.0 * z5["episode_diversity_norm"])),
        ("saved descriptors reproduce the logged D",
         np.allclose(
             [_D(f) for f in z5["token_frequencies"]],
             z5["episode_diversity_bits"])),
        ("token frequency rows sum to 1",
         np.allclose(z5["token_frequencies"].sum(axis=2), 1.0)),
        ("D within its hard bound log2 3",
         z5["episode_diversity_bits"].max() <= np.log2(3) + 1e-12),
        ("D_norm within [0, 1]",
         (z5["episode_diversity_norm"].min() >= -1e-12)
         and (z5["episode_diversity_norm"].max() <= 1 + 1e-12)),
        ("bonus lands on the TERMINAL transition only "
         "(discounted share == gamma^(T-1))",
         _terminal_only(z0, z5)),
        ("permutation null was evaluated",
         len(z5["permutation_null_values"]) > 0),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)

    print()
    print("=" * 68)
    print("TEST 4  the augmented critic at beta = 0 "
          "(the capacity control)")
    print("=" * 68)

    shutil.rmtree(os.path.join(WORK, "results"))
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0"], WARMUP_EPISODES)
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0", "--augment-critic"],
          WARMUP_EPISODES)

    ok &= compare("beta_0", "beta_0_augcritic",
                  "--beta 0 --augment-critic vs --beta 0")

    za = load("beta_0_augcritic")

    checks4 = [
        ("shaped == unshaped exactly at beta = 0 (no bonus at all)",
         np.array_equal(za["episode_shaped_rewards"],
                        za["episode_rewards"])),
        ("D is still computed and logged with the bonus off",
         len(za["episode_diversity_norm"]) == WARMUP_EPISODES),
        ("token frequency rows still sum to 1",
         np.allclose(za["token_frequencies"].sum(axis=2), 1.0)),
        ("no NaN or inf anywhere in the augmented run",
         all(bool(np.all(np.isfinite(v))) for v in za.values()
             if np.issubdtype(np.asarray(v).dtype, np.floating))),
    ]

    import torch
    from models.critic import CentralizedCritic

    c_plain = CentralizedCritic(3, 18, 5, hidden_dim=128, context_dim=0)
    c_aug = CentralizedCritic(3, 18, 5, hidden_dim=128, context_dim=9)

    obs = torch.zeros(4, 3, 18)
    act = torch.zeros(4, 3, 5)
    ctx = torch.zeros(4, 9)

    def _raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    checks4 += [
        ("unaugmented critic input width is 3*18 + 3*5 = 69",
         c_plain.network[0].in_features == 69),
        ("augmented critic input width is 69 + 9 = 78",
         c_aug.network[0].in_features == 78),
        ("augmented critic REFUSES a missing context",
         _raises(lambda: c_aug(obs, act, None))),
        ("unaugmented critic REFUSES a context it was not built for",
         _raises(lambda: c_plain(obs, act, ctx))),
    ]

    for label, passed in checks4:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)

    shutil.rmtree(os.path.join(WORK, "results"))
    train(os.path.join("exp3", "train_exp3.py"),
          ["--beta", "0", "--alpha", "0", "--augment-critic"],
          WARMUP_EPISODES)
    again = load("beta_0_augcritic")
    same = all(np.array_equal(za[k], again[k])
               for k in sorted(set(za) & set(again)))
    print(f"  [{'PASS' if same else 'FAIL'}] augmented run is "
          f"deterministic at a fixed seed")
    ok &= bool(same)

    print()
    print("=" * 68)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED -- do not launch")
    print("=" * 68)
    return 0 if ok else 1


def _D(f):
    from token_diversity import token_diversity
    return token_diversity(f)


def _terminal_only(z0, z5, gamma=0.95, T=25):
    d = z5["episode_diversity_norm"]
    keep = d > 0
    if not keep.any():
        return False
    ratio = (z5["episode_discounted"][keep]
             - z0["episode_discounted"][keep]) / (5.0 * d[keep])
    return bool(np.allclose(ratio, gamma ** (T - 1), atol=1e-9))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
