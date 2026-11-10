import os
import sys
import json
import shutil
import subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "A1r"))

from voronoi_gate import OwnershipGate, fit_log_z, LN10

SEED = 42
EPISODES = 250
ALPHA = 0.01
COMMON = ["--seed", str(SEED), "--batch-size", "256",
          "--update-every", "2", "--print-every", "100000"]

WORK = os.path.join(HERE, ".a1r_reduction_test")
CONST = os.path.join("results", "a1r_pilot", "a1r_constants.json")


def build_sandbox():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(os.path.join(WORK, "exp2"))
    os.makedirs(os.path.join(WORK, "A1r"))
    os.makedirs(os.path.join(WORK, "results", "a1r_pilot"))

    for name in os.listdir(HERE):
        src = os.path.join(HERE, name)
        if (name.endswith(".py") and os.path.isfile(src)) or name == "models":
            os.symlink(src, os.path.join(WORK, name))
    os.symlink(os.path.join(HERE, "exp2", "train_exp2.py"),
               os.path.join(WORK, "exp2", "train_exp2.py"))
    for name in ("train_a1r.py", "maddpg_system_a1r.py", "voronoi_gate.py"):
        os.symlink(os.path.join(HERE, "A1r", name),
                   os.path.join(WORK, "A1r", name))

    real = os.path.join(HERE, CONST)
    if os.path.exists(real):
        shutil.copy(real, os.path.join(WORK, CONST))
        return True
    return False


def train(script, extra):
    cmd = ([sys.executable, os.path.join(WORK, script)] + extra
           + COMMON + ["--episodes", str(EPISODES)])
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def load(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(path, f))
        elif f.endswith(".npz"):
            z = np.load(os.path.join(path, f))
            for k in z.files:
                out[f"{f[:-4]}::{k}"] = z[k]
    return out


def rdir(experiment):
    return os.path.join(WORK, "results", experiment, f"seed_{SEED}")


def compare(a, b, label, skip=()):
    A, B = load(a), load(b)
    shared = sorted((set(A) & set(B)) - set(skip))
    bad = [k for k in shared
           if A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
    ok = not bad
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: "
          f"{len(shared) - len(bad)}/{len(shared)} arrays bit-identical")
    for k in bad:
        x, y = A[k], B[k]
        print(f"          differs: {k}   max|d| = "
              f"{np.abs(np.asarray(x, float) - np.asarray(y, float)).max():.3e}")
    return ok


def main():
    have_constants = build_sandbox()
    ok = True

    print("=" * 70)
    print("A1r REDUCTION CHECKS")
    print("=" * 70)
    print(f"  sandbox   {WORK}")
    print(f"  {EPISODES} episodes, seed {SEED}, alpha {ALPHA}")
    print()

    print("R1  --gate none reproduces Experiment 2 bit-identically")
    train("exp2/train_exp2.py", ["--alpha", str(ALPHA)])
    train("A1r/train_a1r.py", ["--alpha", str(ALPHA), "--gate", "none"])
    ok &= compare(rdir(f"alpha_{ALPHA:g}"), rdir("a1r_none"),
                  "Exp2 vs A1r --gate none")
    print("      -> the A1r copy of maddpg_system.py IS the parent when")
    print("         the gate is off, so the copy introduces no drift.")
    print()

    if not have_constants:
        print("R2-R5  SKIPPED: results/a1r_pilot/a1r_constants.json not "
              "found.\n       Run  python A1r/a1r_pilot.py  first.")
        raise SystemExit(0 if ok else 1)

    const = json.load(open(os.path.join(WORK, CONST)))
    kappa, log_z = const["kappa"], const["log_z"]
    print(f"      frozen constants: kappa = {kappa:.6f}, "
          f"log Z = {log_z:.6f}")
    print()

    print("R2  determinism at fixed seed")
    train("A1r/train_a1r.py", ["--alpha", str(ALPHA), "--gate", "both"])
    snap = os.path.join(WORK, "_run1")
    shutil.copytree(rdir("a1r_both"), snap)
    shutil.rmtree(rdir("a1r_both"))
    train("A1r/train_a1r.py", ["--alpha", str(ALPHA), "--gate", "both"])
    ok &= compare(snap, rdir("a1r_both"), "A1r run 1 vs run 2")
    print()

    gp = os.path.join(HERE, "results", "a1r_pilot", "gaps.npz")
    if not os.path.exists(gp):
        print("R3-R5  SKIPPED: results/a1r_pilot/gaps.npz not found.")
        raise SystemExit(0 if ok else 1)

    G = torch.as_tensor(np.load(gp)["G"].astype(np.float64))
    gate = OwnershipGate(ALPHA, kappa, log_z, mode=const["mode"],
                         clamp_ratio=const["clamp_ratio"])
    log_a = (np.log(ALPHA)
             + torch.nn.functional.logsigmoid(G / kappa).numpy() - log_z)
    a_eff = np.exp(log_a)

    print("R3  geometric mean of alpha_eff equals alpha")
    geo = float(np.exp(log_a.mean()))
    r3 = abs(geo / ALPHA - 1.0) < 1e-9
    ok &= r3
    print(f"  [{'PASS' if r3 else 'FAIL'}] geometric mean = {geo:.10f}  "
          f"(alpha = {ALPHA})   relative deviation "
          f"{abs(geo / ALPHA - 1.0):.2e}")
    print(f"         arithmetic mean = {a_eff.mean():.10f}, above it by "
          f"Jensen -- that gap is what arithmetic normalisation would "
          f"have spent")
    print()

    print("R3b applied (clamped) geometric mean of alpha_eff is within 2% "
          "of alpha")
    cr = const["clamp_ratio"]
    log_a_applied = (np.clip(log_a, np.log(ALPHA / cr), np.log(ALPHA * cr))
                     if cr is not None else log_a)
    geo_ap = float(np.exp(log_a_applied.mean()))
    r3b = abs(geo_ap / ALPHA - 1.0) < 0.02
    ok &= r3b
    print(f"  [{'PASS' if r3b else 'FAIL'}] applied geometric mean = "
          f"{geo_ap:.6f}  = {geo_ap / ALPHA:.4f} x alpha   "
          f"({100 * np.mean(log_a_applied != log_a):.2f}% of samples "
          f"clamped)")
    print()

    print("R4  no reference sample lands outside the actor's clamps")
    sig_a1 = const["reference"]["a1_mean_sigma"]
    clamped = np.clip(a_eff, ALPHA / const["clamp_ratio"],
                      ALPHA * const["clamp_ratio"])
    sig = sig_a1 * np.sqrt(clamped / ALPHA)
    r4 = bool((sig > 0.0067).all() and (sig < 7.3891).all())
    ok &= r4
    print(f"  [{'PASS' if r4 else 'FAIL'}] target sigma in "
          f"[{sig.min():.4f}, {sig.max():.4f}]   "
          f"actor bounds [0.0067, 7.3891]")
    print(f"         {100 * np.mean(a_eff != clamped):.2f}% of samples hit "
          f"the alpha_eff clamp before the sigma clamp could bind")
    print()

    print("R5  kappa -> infinity collapses back onto A1")
    big = 1e9
    huge = OwnershipGate(ALPHA, big, fit_log_z(G, big), mode=const["mode"])
    lw = torch.nn.functional.logsigmoid(G / big).numpy()
    dev = float(np.abs(np.exp(np.log(ALPHA) + lw - fit_log_z(G, big))
                       / ALPHA - 1.0).max())
    r5 = dev < 1e-8
    ok &= r5
    print(f"  [{'PASS' if r5 else 'FAIL'}] max relative deviation from "
          f"alpha = {dev:.2e}")
    print("         w -> 1/2 everywhere and Z -> 1/2, so the ratio is 1.")
    print("         Exact bit-identity is R1's job, via --gate none.")

    print()
    print("=" * 70)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
