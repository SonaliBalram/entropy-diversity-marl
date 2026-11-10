# A1r pilot: freeze the gate constants from trained A1 states and give a go / no-go verdict.

import os
import sys
import json
import argparse
import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from voronoi_gate import (
    gaps, combined_gap, fit_kappa, fit_log_z, OwnershipGate, LN10,
)

MIN_INTERDECILE_G = 0.05
MIN_DISPOSSESSED  = 0.02


# Roll out a policy and keep the observations at which actions were taken.
def collect(env, maddpg, layout_seeds, deterministic, torch_seed):
    torch.manual_seed(torch_seed)
    obs_all, sig_all, rets = [], [], []

    for layout_seed in layout_seeds:
        observations = env.reset(seed=layout_seed)
        total = 0.0
        while True:
            obs_all.append(np.asarray(observations, dtype=np.float64))

            with torch.no_grad():
                dist = maddpg.actor.get_distribution(
                    torch.as_tensor(observations, dtype=torch.float32)
                )
                sig_all.append(
                    dist.scale.log().mean(-1).exp().cpu().numpy()
                )

            actions = maddpg.act(observations, deterministic=deterministic)
            observations, reward, term, trunc, _ = env.step(actions)
            total += reward
            if term or trunc:
                break
        rets.append(total)

    return (np.concatenate(obs_all, axis=0),
            np.concatenate(sig_all, axis=0),
            np.asarray(rets))


# Slope, intercept and correlation of a least-squares line.
def ols(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float((xc * xc).sum())
    if denom <= 0.0:
        return 0.0, float(y.mean()), 0.0
    slope = float((xc * yc).sum() / denom)
    r = float((xc * yc).sum() /
              np.sqrt(denom * float((yc * yc).sum()) + 1e-300))
    return slope, float(y.mean() - slope * x.mean()), r


# Fit kappa and log Z, write the constants file and the pilot report.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints/alpha_0.01",
                    help="the A1 arm; A1r is budget-matched to it")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--rungs", type=int, nargs="+",
                    default=[2000, 10000, 20000, 30000, 40000, 50000],
                    help="checkpoint episodes spanning the trajectory")
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--layout-offset", type=int, default=900_000_000,
                    help="same held-out block as evaluate_baseline.py")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--mode", default="both",
                    choices=["both", "ownership", "landmark"])
    ap.add_argument("--decades", type=float, default=1.0)
    ap.add_argument("--clamp-ratio", type=float, default=10.0)
    ap.add_argument("--deterministic", action="store_true",
                    help="reference from the deterministic pass instead")
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=25)
    ap.add_argument("--out-dir", default="results/a1r_pilot")
    args = ap.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    # Print a line and keep it for the report.
    def say(s=""):
        print(s)
        lines.append(s)

    layout_seeds = [args.layout_offset + i for i in range(args.layouts)]

    say("=" * 72)
    say("A1r PILOT -- reference set, frozen constants, kill switch")
    say("=" * 72)
    say(f"reference arm   {args.ckpt_root}")
    say(f"seeds           {args.seeds}")
    say(f"rungs           {args.rungs}")
    say(f"layouts         {args.layouts} held-out "
        f"({args.layout_offset}+i)")
    say(f"pass            "
        f"{'deterministic' if args.deterministic else 'stochastic'}")
    say("")

    obs_parts, sig_parts, ret_parts, tag_parts = [], [], [], []

    for seed in args.seeds:
        for ep in args.rungs:
            ckpt = os.path.join(PROJECT_ROOT, args.ckpt_root,
                                f"seed_{seed}", f"ckpt_ep{ep}.pt")
            if not os.path.exists(ckpt):
                say(f"  ! missing, skipped: {ckpt}")
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
            maddpg.actor.eval()
            maddpg.critic.eval()

            o, s, r = collect(env, maddpg, layout_seeds,
                              args.deterministic, seed)
            env.close()

            obs_parts.append(o)
            sig_parts.append(s)
            ret_parts.append(r)
            tag_parts.append(np.full(o.shape[0], ep, dtype=np.int64))

            say(f"  seed {seed:>3}  ep{ep:<6}  "
                f"{o.shape[0]:>7} samples   "
                f"sigma {s.mean():.4f}   return {r.mean():8.3f}")

    if not obs_parts:
        raise SystemExit("no checkpoints found -- check --ckpt-root")

    obs = np.concatenate(obs_parts, axis=0)
    sig = np.concatenate(sig_parts, axis=0)
    rung = np.concatenate(tag_parts, axis=0)
    say("")
    say(f"reference set: {obs.shape[0]} (state, agent) samples")

    m_t, g_t = gaps(torch.as_tensor(obs))
    m = m_t.numpy()
    g = g_t.numpy()
    G = combined_gap(m_t, g_t, args.mode).numpy()

    # Quantile of an array.
    def q(a, p):
        return float(np.quantile(a, p))

    say("")
    say("-" * 72)
    say("GAP DISTRIBUTION (reward units)")
    say("-" * 72)
    say(f"{'':22s}{'q10':>9}{'q25':>9}{'median':>9}"
        f"{'q75':>9}{'q90':>9}{'mean':>9}")
    for name, a in (("landmark margin m", m),
                    ("ownership gap  g", g),
                    ("combined gap   G", G)):
        say(f"{name:22s}{q(a,.10):9.4f}{q(a,.25):9.4f}{q(a,.50):9.4f}"
            f"{q(a,.75):9.4f}{q(a,.90):9.4f}{a.mean():9.4f}")

    inter_g = q(G, .90) - q(G, .10)
    say("")
    say(f"interdecile range of G: {inter_g:.4f} reward units")

    say("")
    say("-" * 72)
    say("FROZEN CONSTANTS")
    say("-" * 72)

    kappa = fit_kappa(G, decades=args.decades)
    log_z = fit_log_z(G, kappa)
    gate = OwnershipGate(args.alpha, kappa, log_z, mode=args.mode,
                         clamp_ratio=args.clamp_ratio)

    closed = (q(G, .90) - q(G, .10)) / (2 * args.decades * LN10)
    say(f"kappa  = {kappa:.6f}      "
        f"(closed-form approximation {closed:.6f})")
    say(f"log Z  = {log_z:.6f}      Z = {np.exp(log_z):.6f}")
    say(f"mode   = {args.mode}    decades = {args.decades}    "
        f"clamp = alpha x [1/{args.clamp_ratio:g}, {args.clamp_ratio:g}]")

    d = gate.diagnostics(torch.as_tensor(obs))
    say("")
    say(f"alpha_eff  geometric mean {d['alpha_eff_geomean']:.6f}  "
        f"(target {args.alpha:g})")
    say(f"           arithmetic     {d['alpha_eff_mean']:.6f}  "
        f"(above it, by Jensen)")
    say(f"           q10 {d['alpha_eff_q10']:.5f}   "
        f"q90 {d['alpha_eff_q90']:.5f}   "
        f"spread {d['interdecile_log_alpha'] / LN10:.4f} decades")
    say(f"           clamped {100 * d['frac_clamped']:.2f}% of samples")

    sig_a1 = float(sig.mean())
    s_lo = sig_a1 * np.sqrt(d["alpha_eff_q10"] / args.alpha)
    s_hi = sig_a1 * np.sqrt(d["alpha_eff_q90"] / args.alpha)
    s_min = sig_a1 * np.sqrt(1.0 / args.clamp_ratio)
    s_max = sig_a1 * np.sqrt(args.clamp_ratio)
    say("")
    say(f"predicted sigma* bulk  [{s_lo:.4f}, {s_hi:.4f}]   "
        f"pinned at A1's {sig_a1:.4f}")
    say(f"predicted sigma* clamp [{s_min:.4f}, {s_max:.4f}]   "
        f"actor bounds [0.0067, 7.3891]")
    say(f"R4 (clamps respected): "
        f"{'PASS' if 0.0067 < s_min and s_max < 7.3891 else 'FAIL'}")

    say("")
    say("-" * 72)
    say("ALLOCATION QUADRANTS   (boundary band = one kappa)")
    say("-" * 72)

    owns_nothing = g >= 0.0
    contested = np.abs(g) < kappa
    secure = g < -kappa
    target_vague = m < kappa

    # Percentage of True values in a mask.
    def pc(mask):
        return 100.0 * float(mask.mean())

    say(f"{'':34s}{'target vague':>16}{'target clear':>16}")
    for name, own in (("owns nothing      (g >= 0)", owns_nothing),
                      ("contested         (|g| < k)", contested),
                      ("secure owner      (g < -k)", secure)):
        say(f"{name:34s}{pc(own & target_vague):15.2f}%"
            f"{pc(own & ~target_vague):15.2f}%")

    # Mean gate weight over the states in a mask.
    def mean_w(mask):
        if not mask.any():
            return float("nan")
        return float(np.exp(
            -np.logaddexp(0.0, -G[mask] / kappa)).mean())

    binds_landmark = (-m) > g
    say("")
    say("WHICH DIAGRAM IS DECIDING   (G = max(g, -m))")
    say(f"  landmark-sited  (G = -m)   {100 * binds_landmark.mean():6.2f}%")
    say(f"  agent-sited     (G =  g)   "
        f"{100 * (~binds_landmark).mean():6.2f}%")
    rho = float(np.corrcoef(g, -m)[0, 1])
    say(f"  corr(g, -m) = {rho:+.4f}   (R^2 = {rho ** 2:.3f})")
    say("  Both branches well away from 0% and 100% is what 'combines two")
    say("  Voronoi diagrams' has to mean in practice; if either were near 0")
    say("  the arm would really be using one diagram and should be")
    say("  simplified to it.")
    say("")
    say("  The two gaps are CORRELATED -- in trained play an agent with a")
    say("  clear nearest landmark tends also to own it securely -- so they")
    say("  are not independent evidence.  What the landmark margin cannot")
    say("  supply is the SIGN, and the sign is what creates the division of")
    say("  labour.  The check below asks whether m alone could stand in for")
    say("  ownership:")
    base = 100.0 * float((~owns_nothing).mean())
    accs = [(100.0 * float(((m < t) == owns_nothing).mean()), t)
            for t in np.linspace(0.0, 2.0, 201)]
    acc, thr = max(accs)
    say(f"    best single m-threshold classifier for 'owns nothing': "
        f"{acc:.1f}% at m < {thr:.2f}")
    say(f"    base rate of always answering 'owns something':        "
        f"{base:.1f}%")
    if acc - base < 1.0:
        say("    -> m carries no usable information about ownership.  The")
        say("       agent-sited diagram is not redundant however correlated")
        say("       the continuous gaps are.")
    else:
        say("    -> m does partially predict ownership; consider whether the")
        say("       agent-sited term is earning its place.")

    say("")
    say(f"mean gate w:  owns nothing {mean_w(owns_nothing):.4f}"
        f"   contested {mean_w(contested):.4f}"
        f"   secure owner {mean_w(secure):.4f}")

    say("")
    say("-" * 72)
    say("COST OF FREEZING Z   (realised geometric mean of alpha_eff)")
    say("-" * 72)
    say("Z is frozen so that the gate stays a fixed function of the")
    say("observation.  The price is that the realised budget drifts as")
    say("training moves the state distribution.  That drift is not a bug to")
    say("be corrected online -- correcting it would reintroduce the moving")
    say("scale the freeze exists to prevent -- but it MUST be reported,")
    say("because a realised budget away from alpha means A1r is partly an")
    say("'A1 at a different alpha' comparison.")
    say("")
    say(f"  {'rung':>8}{'geo mean':>12}{'/ alpha':>10}"
        f"{'sigma factor':>14}{'spread (dec)':>14}")
    with torch.no_grad():
        for ep in sorted(set(rung.tolist())):
            sel = rung == ep
            la = (np.log(args.alpha)
                  + torch.nn.functional.logsigmoid(
                      torch.as_tensor(G[sel]) / kappa).numpy()
                  - log_z)
            gm = float(np.exp(la.mean()))
            spread = (float(np.quantile(la, 0.90))
                      - float(np.quantile(la, 0.10))) / LN10
            say(f"  {ep:>8}{gm:12.6f}{gm / args.alpha:10.3f}"
                f"{np.sqrt(gm / args.alpha):14.3f}{spread:14.3f}")
    say("")
    say("  sigma factor is sqrt of the alpha ratio, since sigma* propto")
    say("  sqrt(alpha_eff).  Compare it against the seed-to-seed spread of")
    say("  A1's own sigma before deciding whether the drift matters.")
    say("  If 'spread' collapses towards 0 the allocation has flattened and")
    say("  A1r has quietly become A1.")

    say("")
    say("-" * 72)
    say("tau BASELINE  (A1's own sigma head, against the frozen gate)")
    say("-" * 72)

    with torch.no_grad():
        half_log_a = 0.5 * (
            np.log(args.alpha)
            + torch.nn.functional.logsigmoid(
                torch.as_tensor(G) / kappa).numpy()
            - log_z
        )
    slope, _, r = ols(half_log_a, np.log(sig))
    say(f"slope of log sigma on (1/2) log alpha_eff : "
        f"{slope:+.4f}   (r = {r:+.4f})")
    say("")
    say("  A1 trained at constant alpha, so it had no reason to track "
        "this gate.")
    say("  Whatever slope appears here is background correlation "
        "between the sigma")
    say("  head and the geometry.  A1r's tau is read against THIS "
        "number, not zero:")
    say(f"      tau = (A1r slope - {slope:+.4f}) / (1 - {slope:+.4f})")
    say("  tau ~ 1 -> pointwise cancellation, predicted null.")
    say("  tau ~ 0 -> manipulation not delivered, INCONCLUSIVE not null.")
    say("  0 < tau < 1 -> the only regime in which A1r can move "
        "anything.")

    say("")
    say(f"  {'rung':>8}{'slope':>10}{'r':>9}{'mean sigma':>13}")
    for ep in sorted(set(rung.tolist())):
        sel = rung == ep
        sl, _, rr = ols(half_log_a[sel], np.log(sig[sel]))
        say(f"  {ep:>8}{sl:+10.4f}{rr:+9.4f}{sig[sel].mean():13.4f}")

    say("")
    say("=" * 72)
    say("KILL-SWITCH VERDICT  (rules fixed before the numbers were seen)")
    say("=" * 72)

    frac_disp = float(owns_nothing.mean())
    verdict, reasons = "RUN A1r AS DESIGNED", []

    if inter_g < MIN_INTERDECILE_G:
        verdict = "DO NOT RUN A1r"
        reasons.append(
            f"interdecile G = {inter_g:.4f} < {MIN_INTERDECILE_G} "
            f"reward units: the gap is near-constant, so there is no "
            f"allocation to make and the arm would be A1 with extra steps")
    elif frac_disp < MIN_DISPOSSESSED:
        verdict = "RUN, BUT FALL BACK TO mode='landmark'"
        reasons.append(
            f"only {100 * frac_disp:.2f}% of samples have g >= 0 "
            f"(< {100 * MIN_DISPOSSESSED:.0f}%): the agent-sited diagram "
            f"is inert on this task, so the ownership term does no work")
    else:
        reasons.append(
            f"interdecile G = {inter_g:.4f} reward units and "
            f"{100 * frac_disp:.2f}% of samples own nothing: both "
            f"diagrams carry usable signal")

    say(f"  {verdict}")
    for r_ in reasons:
        say(f"    - {r_}")

    const = dict(
        gate.to_dict(),
        decades=args.decades,
        interdecile_G=inter_g,
        frac_owns_nothing=frac_disp,
        tau_baseline_slope=slope,
        reference=dict(
            ckpt_root=args.ckpt_root, seeds=args.seeds, rungs=args.rungs,
            layouts=args.layouts, layout_offset=args.layout_offset,
            deterministic=bool(args.deterministic),
            n_samples=int(obs.shape[0]),
            a1_mean_sigma=sig_a1,
        ),
        verdict=verdict,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    cpath = os.path.join(out_dir, "a1r_constants.json")
    with open(cpath, "w") as f:
        json.dump(const, f, indent=2)

    np.savez_compressed(
        os.path.join(out_dir, "gaps.npz"),
        m=m.astype(np.float32), g=g.astype(np.float32),
        G=G.astype(np.float32), sigma=sig.astype(np.float32),
        rung=rung,
    )
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    say("")
    say(f"constants -> {cpath}")
    say("             train_a1r.py reads this file and stores its "
        "contents in every checkpoint.")
    say(f"raw gaps  -> {os.path.join(out_dir, 'gaps.npz')}")


if __name__ == "__main__":
    main()
