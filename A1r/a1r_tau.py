"""
a1r_tau.py  --  the tracking coefficient.

tau answers the question A1r actually turns on: did the policy's sigma
head TRACK the state-dependent target the gate asked for?

The gate sets a target

    sigma*(o) propto sqrt(alpha_eff(o))   =>   log sigma* = (1/2) log
                                               alpha_eff(o) + constant

so a policy that tracks perfectly has a slope of 1 when its measured
log sigma is regressed on (1/2) log alpha_eff.  Define

    tau = (slope_measured - slope_baseline) / (1 - slope_baseline)

The baseline correction matters.  A1 trained at constant alpha, so it
had no reason to track this gate, but its sigma head may still correlate
with the geometry by accident.  a1r_pilot.py measures that background
slope; tau is read against it, not against zero.

INTERPRETATION -- fixed in advance, because this is exactly what went
wrong with A2c's pre-registered signature:

    tau ~ 1      the policy equilibrates pointwise, the Exchange Rate
                 for a Nat result applies at every state, and A1r is
                 predicted to be a NULL for the same reason Exp 2 was.
    tau ~ 0      the manipulation was never delivered.  This is
                 INCONCLUSIVE, not a null: nothing was tested.
    0 < tau < 1  partial tracking -- the ONLY regime in which A1r can
                 move the outcome measure at all.

Both endpoints predict nothing, so tau MUST be reported before and
alongside any Delta D, never after it.

    python A1r/a1r_tau.py --ckpt-root checkpoints/a1r_both

Outputs results/a1r_tau/tau_<arm>.txt and .npz
"""

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

from environment import MPEEnvironment            # noqa: E402
from maddpg_system_a1r import MADDPGSystem        # noqa: E402
from voronoi_gate import gaps, combined_gap       # noqa: E402
from a1r_pilot import collect, ols                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints/a1r_both")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--rungs", type=int, nargs="+",
                    default=[2000, 10000, 20000, 30000, 40000, 50000])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--layout-offset", type=int, default=900_000_000)
    ap.add_argument("--constants",
                    default="results/a1r_pilot/a1r_constants.json")
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
    base = const["tau_baseline_slope"]

    layout_seeds = [args.layout_offset + i for i in range(args.layouts)]

    say("=" * 72)
    say(f"TRACKING COEFFICIENT tau   --   arm {arm}")
    say("=" * 72)
    say(f"frozen gate     kappa {kappa:.6f}   log Z {log_z:.6f}   "
        f"mode {mode}")
    say(f"A1 baseline slope {base:+.4f}   (from the pilot)")
    say("")
    say(f"  {'seed':>6}{'rung':>8}{'slope':>10}{'tau':>9}{'r':>9}"
        f"{'sigma':>10}{'spread':>9}")

    rows = []
    for seed in args.seeds:
        for ep in args.rungs:
            ckpt = os.path.join(PROJECT_ROOT, args.ckpt_root,
                                f"seed_{seed}", f"ckpt_ep{ep}.pt")
            if not os.path.exists(ckpt):
                continue
            c = torch.load(ckpt, map_location="cpu", weights_only=False)
            env = MPEEnvironment(n_agents=args.n_agents,
                                 max_cycles=args.max_cycles)
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

            obs, sig, _ = collect(env, maddpg, layout_seeds,
                                  args.deterministic, seed)
            env.close()

            m, g = gaps(torch.as_tensor(obs))
            G = combined_gap(m, g, mode)
            half = 0.5 * (np.log(alpha)
                          + torch.nn.functional.logsigmoid(
                              G / kappa).numpy() - log_z)
            slope, _, r = ols(half, np.log(sig))
            tau = (slope - base) / (1.0 - base)
            spread = (np.quantile(half, .90) - np.quantile(half, .10)) * 2
            say(f"  {seed:>6}{ep:>8}{slope:+10.4f}{tau:+9.4f}{r:+9.4f}"
                f"{sig.mean():10.4f}{spread / np.log(10):9.3f}")
            rows.append((seed, ep, slope, tau, r, float(sig.mean())))

    if not rows:
        raise SystemExit(f"no checkpoints under {args.ckpt_root}")

    arr = np.array([[a, b, c_, d, e, f] for a, b, c_, d, e, f in rows])
    say("")
    say("-" * 72)

    # headline: tau at the final rung, paired across seeds
    final = max(args.rungs)
    sel = arr[:, 1] == final
    if sel.any():
        t = arr[sel, 3]
        say(f"tau at episode {final}, across {sel.sum()} seeds: "
            f"{t.mean():+.4f} +/- {t.std(ddof=1) if len(t) > 1 else 0:.4f}")
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

    with open(os.path.join(out_dir, f"tau_{arm}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, f"tau_{arm}.npz"),
             rows=arr,
             columns=np.array(["seed", "rung", "slope", "tau", "r",
                               "sigma"]))
    say(f"\nwritten -> {os.path.join(out_dir, f'tau_{arm}.txt')}")


if __name__ == "__main__":
    main()
