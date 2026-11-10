import os
import sys
import json
import argparse

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import MPEEnvironment
from maddpg_system_a1r import MADDPGSystem
from voronoi_gate import gaps, combined_gap, OwnershipGate
from a1r_pilot import collect, ols


def half_log_alpha(G, alpha, kappa, log_z, clamp_ratio):
    G = torch.as_tensor(np.asarray(G, dtype=np.float64))
    log_a = (np.log(alpha)
             + torch.nn.functional.logsigmoid(G / kappa).numpy() - log_z)
    if clamp_ratio is not None:
        log_a = np.clip(log_a, np.log(alpha / clamp_ratio),
                        np.log(alpha * clamp_ratio))
    return 0.5 * log_a


def half_log_alpha_unclamped(G, alpha, kappa, log_z):
    return half_log_alpha(G, alpha, kappa, log_z, None)


def load_system(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    maddpg = MADDPGSystem(
        n_agents=c["n_agents"],
        observation_dim=c["observation_dim"],
        action_dim=c["action_dim"],
        hidden_dim=c["hidden_dim"],
        gamma=c["gamma"], tau=c["tau"],
        twin_critics=c.get("twin_critics", True),
        context_dim=c.get("context_dim", 0),
        device="cpu",
    )
    maddpg.load(ckpt)
    maddpg.actor.eval(); maddpg.critic.eval()
    return maddpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints/a1r_both")
    ap.add_argument("--baseline-ckpt-root", default=None,
                    help="A1 checkpoints (e.g. checkpoints/alpha_0.01). "
                         "If given, tau is ALSO computed against A1's slope "
                         "at the same seed and rung (tau_paired).")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--rungs", type=int, nargs="+",
                    default=[2000, 10000, 20000, 30000, 40000, 50000])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--layout-offset", type=int, default=900_000_000)
    ap.add_argument("--constants",
                    default="results/a1r_pilot/a1r_constants.json")
    ap.add_argument("--gaps", default="results/a1r_pilot/gaps.npz",
                    help="the pilot's A1 reference states (per-rung baseline)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=25)
    ap.add_argument("--out-dir", default="results/a1r_tau")
    args = ap.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arm = os.path.basename(args.ckpt_root.rstrip("/"))
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    const = json.load(open(os.path.join(PROJECT_ROOT, args.constants)))
    kappa, log_z = const["kappa"], const["log_z"]
    alpha, mode = const["alpha"], const["mode"]
    clamp = const.get("clamp_ratio")
    base_pooled = const["tau_baseline_slope"]

    gate = OwnershipGate(alpha, kappa, log_z, mode=mode, clamp_ratio=clamp)

    gz = np.load(os.path.join(PROJECT_ROOT, args.gaps))
    ref_half = half_log_alpha(gz["G"], alpha, kappa, log_z, clamp)
    ref_logsig = np.log(gz["sigma"].astype(np.float64))
    base_rung = {}
    for ep in args.rungs:
        sel = gz["rung"] == ep
        if not sel.any():
            raise SystemExit(
                f"rung {ep} is not in {args.gaps}; the per-rung baseline "
                f"needs A1 reference states at every rung analysed")
        base_rung[ep] = ols(ref_half[sel], ref_logsig[sel])[0]

    layout_seeds = [args.layout_offset + i for i in range(args.layouts)]

    say("=" * 78)
    say(f"TRACKING COEFFICIENT tau (corrected)   --   arm {arm}")
    say("=" * 78)
    say(f"frozen gate     kappa {kappa:.6f}   log Z {log_z:.6f}   "
        f"mode {mode}   clamp alpha x [1/{clamp:g}, {clamp:g}]")
    say("regressor       (1/2) log alpha_eff AS APPLIED (clamped)")
    say("baseline        A1's slope at the same rung, from the pilot "
        "(tau)")
    if args.baseline_ckpt_root:
        say(f"                A1's slope at the same seed AND rung, from "
            f"{args.baseline_ckpt_root} (tau_paired)")
    say(f"tau_orig        pre-registered version: unclamped regressor, "
        f"pooled baseline {base_pooled:+.4f}")
    say("")
    say("A1 baseline slope per rung (clamped regressor):")
    say("  " + "   ".join(f"{ep}: {base_rung[ep]:+.4f}" for ep in args.rungs))
    say("")
    say(f"  {'seed':>6}{'rung':>8}{'slope':>9}{'tau':>9}{'tau_pair':>10}"
        f"{'tau_orig':>10}{'r':>8}{'sigma':>8}{'spread':>8}{'clamp%':>8}")

    rows = []
    for seed in args.seeds:
        for ep in args.rungs:
            ckpt = os.path.join(PROJECT_ROOT, args.ckpt_root,
                                f"seed_{seed}", f"ckpt_ep{ep}.pt")
            if not os.path.exists(ckpt):
                continue
            env = MPEEnvironment(n_agents=args.n_agents,
                                 max_cycles=args.max_cycles)
            obs, sig, _ = collect(env, load_system(ckpt), layout_seeds,
                                  args.deterministic, seed)
            env.close()
            logsig = np.log(sig)

            a_eff = gate(torch.as_tensor(obs)).numpy().astype(np.float64)
            half = 0.5 * np.log(a_eff)

            m, g = gaps(torch.as_tensor(obs))
            G = combined_gap(m, g, mode).numpy()
            assert np.allclose(half, half_log_alpha(G, alpha, kappa, log_z,
                                                    clamp), atol=1e-9)

            slope, _, r = ols(half, logsig)
            b = base_rung[ep]
            tau = (slope - b) / (1.0 - b)

            slope_o, _, _ = ols(half_log_alpha_unclamped(G, alpha, kappa,
                                                         log_z), logsig)
            tau_orig = (slope_o - base_pooled) / (1.0 - base_pooled)

            tau_pair = np.nan
            if args.baseline_ckpt_root:
                bckpt = os.path.join(PROJECT_ROOT, args.baseline_ckpt_root,
                                     f"seed_{seed}", f"ckpt_ep{ep}.pt")
                if os.path.exists(bckpt):
                    env = MPEEnvironment(n_agents=args.n_agents,
                                         max_cycles=args.max_cycles)
                    bobs, bsig, _ = collect(env, load_system(bckpt),
                                            layout_seeds,
                                            args.deterministic, seed)
                    env.close()
                    bhalf = 0.5 * np.log(gate(torch.as_tensor(bobs))
                                         .numpy().astype(np.float64))
                    bp = ols(bhalf, np.log(bsig))[0]
                    tau_pair = (slope - bp) / (1.0 - bp)

            spread = (np.quantile(half, .90) - np.quantile(half, .10)) * 2
            clamped = float(np.mean(
                a_eff <= alpha / clamp * (1 + 1e-9))) if clamp else 0.0

            say(f"  {seed:>6}{ep:>8}{slope:+9.4f}{tau:+9.4f}"
                f"{tau_pair:+10.4f}{tau_orig:+10.4f}{r:+8.3f}"
                f"{sig.mean():8.4f}{spread / np.log(10):8.3f}"
                f"{100 * clamped:8.2f}")
            rows.append((seed, ep, slope, tau, tau_pair, tau_orig, r,
                         float(sig.mean()), spread / np.log(10), clamped))

    if not rows:
        raise SystemExit(f"no checkpoints under {args.ckpt_root}")

    arr = np.array(rows, dtype=np.float64)
    cols = ["seed", "rung", "slope", "tau", "tau_paired", "tau_orig", "r",
            "sigma", "spread_decades", "frac_clamped"]

    say("")
    say("-" * 78)
    say(f"  {'rung':>8}{'tau':>18}{'tau_paired':>18}{'tau_orig':>18}")
    for ep in args.rungs:
        sel = arr[:, 1] == ep
        if not sel.any():
            continue

        def ms(col):
            v = arr[sel, cols.index(col)]
            if np.all(np.isnan(v)):
                return f"{'--':>18}"
            sd = np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else 0.0
            return f"{np.nanmean(v):+10.4f} +/-{sd:.4f}"
        say(f"  {ep:>8}{ms('tau')}{ms('tau_paired')}{ms('tau_orig')}")

    final = max(args.rungs)
    sel = arr[:, 1] == final
    if sel.any():
        t = arr[sel, cols.index("tau")]
        say("")
        say(f"tau at episode {final}, across {sel.sum()} seeds: "
            f"{t.mean():+.4f} +/- {t.std(ddof=1) if len(t) > 1 else 0:.4f}"
            f"   (tau_orig {arr[sel, cols.index('tau_orig')].mean():+.4f})")
        if t.mean() > 0.8:
            verdict = ("tau ~ 1: pointwise cancellation. A1r is predicted "
                       "to be a NULL, for the same reason Exp 2 was.")
        elif t.mean() < 0.2:
            verdict = ("tau ~ 0: the manipulation was NOT delivered. "
                       "INCONCLUSIVE -- nothing was tested. Do not report "
                       "this as a null.")
        else:
            verdict = ("0 < tau < 1: partial tracking. This is the only "
                       "regime in which A1r can move Delta D, so the "
                       "outcome measure is worth reading.")
        say(f"  -> {verdict}")
    say("")
    say("Report this BEFORE and ALONGSIDE any Delta D, never after.")

    stem = f"tau_{arm}_corrected"
    with open(os.path.join(out_dir, f"{stem}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, f"{stem}.npz"),
             rows=arr, columns=np.array(cols),
             baseline_rungs=np.array(args.rungs),
             baseline_slopes=np.array([base_rung[e] for e in args.rungs]))
    say(f"\nwritten -> {os.path.join(out_dir, f'{stem}.txt')}")


if __name__ == "__main__":
    main()
