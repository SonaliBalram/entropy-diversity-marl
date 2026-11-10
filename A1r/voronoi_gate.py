import math

import os as _os
import sys as _sys

PROJECT_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))
)
if PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


LANDMARK_BLOCK    = slice(4, 10)
OTHER_AGENT_BLOCK = slice(10, 14)
N_LANDMARKS = 3
N_RIVALS    = 2

LN10 = math.log(10.0)

MODES = ("both", "ownership", "landmark")


def gaps(obs):
    o = torch.as_tensor(obs)
    lead = o.shape[:-1]

    L = o[..., LANDMARK_BLOCK].reshape(*lead, N_LANDMARKS, 2)
    A = o[..., OTHER_AGENT_BLOCK].reshape(*lead, N_RIVALS, 2)

    rho_self = torch.linalg.vector_norm(L, dim=-1)

    rho_rival = torch.linalg.vector_norm(
        L.unsqueeze(-2) - A.unsqueeze(-3), dim=-1
    )

    c = rho_self - rho_rival.min(dim=-1).values
    g = c.min(dim=-1).values

    top2 = torch.topk(rho_self, 2, dim=-1, largest=False).values
    m = top2[..., 1] - top2[..., 0]

    return m, g


def combined_gap(m, g, mode="both"):
    if mode == "both":
        return torch.maximum(g, -m)
    if mode == "ownership":
        return g
    if mode == "landmark":
        return -m
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def log_weight(G, kappa):
    return F.logsigmoid(torch.as_tensor(G) / kappa)


def fit_kappa(G_ref, decades=1.0, q_lo=0.10, q_hi=0.90):
    g = torch.as_tensor(np.asarray(G_ref, dtype=np.float64)).flatten()
    lo = torch.quantile(g, q_lo).item()
    hi = torch.quantile(g, q_hi).item()

    if hi - lo < 1e-9:
        raise ValueError(
            f"reference gap is near-constant (q{q_hi:.2f} - q{q_lo:.2f} "
            f"= {hi - lo:.3e}).  There is no allocation to make: the "
            f"pilot kill switch has fired and A1r should not be run."
        )

    target = decades * LN10

    def spread(k):
        t = torch.tensor([hi / k, lo / k], dtype=torch.float64)
        lw = F.logsigmoid(t)
        return (lw[0] - lw[1]).item()

    k_lo, k_hi = 1e-8, 1e8
    for _ in range(200):
        mid = math.sqrt(k_lo * k_hi)
        if spread(mid) > target:
            k_lo = mid
        else:
            k_hi = mid

    return math.sqrt(k_lo * k_hi)


def fit_log_z(G_ref, kappa):
    g = torch.as_tensor(np.asarray(G_ref, dtype=np.float64)).flatten()
    return float(log_weight(g, kappa).mean())


class OwnershipGate:
    def __init__(self, alpha, kappa, log_z,
                 mode="both", clamp_ratio=10.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be finite and > 0")

        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.log_z = float(log_z)
        self.mode = mode
        self.clamp_ratio = (None if clamp_ratio is None
                            else float(clamp_ratio))
        self._log_alpha = math.log(self.alpha)

    def __call__(self, obs):
        with torch.no_grad():
            m, g = gaps(obs)
            G = combined_gap(m, g, self.mode)
            log_a = self._log_alpha + log_weight(G, self.kappa) - self.log_z
            a = torch.exp(log_a)
            if self.clamp_ratio is not None:
                a = a.clamp(self.alpha / self.clamp_ratio,
                            self.alpha * self.clamp_ratio)
        return a.detach()

    def diagnostics(self, obs):
        with torch.no_grad():
            m, g = gaps(obs)
            G = combined_gap(m, g, self.mode)
            log_a = self._log_alpha + log_weight(G, self.kappa) - self.log_z
            flat = log_a.flatten().double()
            q10 = torch.quantile(flat, 0.10).item()
            q90 = torch.quantile(flat, 0.90).item()
            if self.clamp_ratio is not None:
                applied = flat.clamp(
                    self._log_alpha - math.log(self.clamp_ratio),
                    self._log_alpha + math.log(self.clamp_ratio))
            else:
                applied = flat
            return {
                "alpha_eff_geomean_applied": float(torch.exp(applied.mean())),
                "alpha_eff_mean_applied": float(torch.exp(applied).mean()),
                "frac_landmark_binds": float(((-m) > g).double().mean()),
                "alpha_eff_geomean": float(torch.exp(flat.mean())),
                "alpha_eff_mean": float(torch.exp(flat).mean()),
                "alpha_eff_q10": math.exp(q10),
                "alpha_eff_q90": math.exp(q90),
                "interdecile_log_alpha": q90 - q10,
                "mean_m": float(m.double().mean()),
                "mean_g": float(g.double().mean()),
                "frac_owns_nothing": float((g >= 0).double().mean()),
                "frac_clamped": float(
                    ((flat < self._log_alpha - math.log(self.clamp_ratio)) |
                     (flat > self._log_alpha + math.log(self.clamp_ratio))
                     ).double().mean()
                ) if self.clamp_ratio is not None else 0.0,
            }

    def to_dict(self):
        return {"alpha": self.alpha, "kappa": self.kappa,
                "log_z": self.log_z, "mode": self.mode,
                "clamp_ratio": self.clamp_ratio}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def fit(cls, obs_ref, alpha, mode="both", decades=1.0,
            clamp_ratio=10.0):
        m, g = gaps(torch.as_tensor(
            np.asarray(obs_ref, dtype=np.float64)))
        G = combined_gap(m, g, mode)
        kappa = fit_kappa(G, decades=decades)
        log_z = fit_log_z(G, kappa)
        return cls(alpha, kappa, log_z, mode=mode,
                   clamp_ratio=clamp_ratio)

    def __repr__(self):
        return (f"OwnershipGate(alpha={self.alpha:g}, "
                f"kappa={self.kappa:.6g}, log_z={self.log_z:.6g}, "
                f"mode={self.mode!r}, clamp_ratio={self.clamp_ratio})")


def synthetic_observations(positions, landmarks):
    positions = np.asarray(positions, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    n = positions.shape[0]
    dim = 4 + 2 * landmarks.shape[0] + 2 * (n - 1) + 2 * (n - 1)
    obs = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        obs[i, 0:2] = 0.0
        obs[i, 2:4] = positions[i]
        obs[i, LANDMARK_BLOCK] = (landmarks - positions[i]).reshape(-1)
        others = [j for j in range(n) if j != i]
        obs[i, OTHER_AGENT_BLOCK] = (
            positions[others] - positions[i]).reshape(-1)
    return obs


def brute_force(positions, landmarks):
    positions = np.asarray(positions, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    n, K = positions.shape[0], landmarks.shape[0]

    rho = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            rho[i, k] = np.linalg.norm(positions[i] - landmarks[k])

    m = np.zeros(n)
    g = np.zeros(n)
    delta = np.zeros(n)
    for i in range(n):
        s = np.sort(rho[i])
        m[i] = s[1] - s[0]
        c = np.array([rho[i, k] - min(rho[j, k] for j in range(n) if j != i)
                      for k in range(K)])
        g[i] = c.min()
        delta[i] = np.maximum(0.0, -c).sum()
    return m, g, delta


def _self_test():
    rng = np.random.default_rng(20260922)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              f"{('   ' + detail) if detail else ''}")

    print("\n1. gate from o_i reproduces ground truth from global positions")
    max_dm = max_dg = 0.0
    for _ in range(400):
        pos = rng.uniform(-1.5, 1.5, size=(3, 2))
        lm = rng.uniform(-1.5, 1.5, size=(3, 2))
        m_t, g_t, _ = brute_force(pos, lm)
        m_o, g_o = gaps(torch.as_tensor(synthetic_observations(pos, lm)))
        max_dm = max(max_dm, float(np.abs(m_o.numpy() - m_t).max()))
        max_dg = max(max_dg, float(np.abs(g_o.numpy() - g_t).max()))
    check("m exact from observation", max_dm < 1e-12, f"max |dm| = {max_dm:.2e}")
    check("g exact from observation", max_dg < 1e-12, f"max |dg| = {max_dg:.2e}")
    print("     -> confirms l_k - x_j = L_k - A_j; the gate needs no "
          "privileged information")

    print("\n2. marginal-contribution identity:  Delta_i = 0  <=>  g_i >= 0")
    bad = 0
    for _ in range(800):
        pos = rng.uniform(-1.5, 1.5, size=(3, 2))
        lm = rng.uniform(-1.5, 1.5, size=(3, 2))
        _, g_t, delta = brute_force(pos, lm)
        if not np.all((delta <= 1e-12) == (g_t >= -1e-12)):
            bad += 1
    check("equivalence holds on every sample", bad == 0,
          f"{bad} violations in 800")

    print("\n3. ownership flips exactly at the perpendicular bisector")
    pos = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 60.0]])
    xs = np.linspace(-0.4, 0.4, 81)
    diff = np.array([
        np.linalg.norm(pos[0] - np.array([x, 0.3]))
        - np.linalg.norm(pos[1] - np.array([x, 0.3]))
        for x in xs
    ])
    lm0 = np.repeat(np.array([[0.0, 0.3]]), 3, axis=0)
    _, g_mid, _ = brute_force(pos, lm0)
    tie = np.isclose(diff, 0.0)
    check("agent 0 owns it strictly left of the bisector",
          bool(np.all(diff[xs < 0] < 0)))
    check("agent 1 owns it strictly right of the bisector",
          bool(np.all(diff[xs > 0] > 0)))
    check("the tie occurs at x = 0 and nowhere else",
          bool(np.count_nonzero(tie) == 1 and abs(xs[tie][0]) < 1e-12),
          f"tie at x = {xs[tie][0]:+.1e}, "
          f"|rho_0 - rho_1| = {abs(diff[tie][0]):.2e}")
    check("g = 0 on the bisector for both contesting agents",
          abs(g_mid[0]) < 1e-12 and abs(g_mid[1]) < 1e-12,
          f"g = ({g_mid[0]:.2e}, {g_mid[1]:.2e})")
    check("w = 1/2 on the bisector",
          abs(float(torch.exp(log_weight(torch.tensor([0.0]), 0.3))) - 0.5)
          < 1e-12)

    print("\n4. signed gate gives a division of labour, |g| would not")
    pos = np.array([[-0.8, 0.0], [0.5, 0.0], [0.0, 60.0]])
    lm = np.repeat(np.array([[0.1, 0.0]]), 3, axis=0)
    _, g_t, _ = brute_force(pos, lm)
    check("g_0 = -g_1 exactly", abs(g_t[0] + g_t[1]) < 1e-12,
          f"g = ({g_t[0]:+.4f}, {g_t[1]:+.4f})")
    w = torch.exp(log_weight(torch.as_tensor(g_t[:2]), 0.25))
    check("w_0 + w_1 = 1 (one commits, the other explores)",
          abs(float(w.sum()) - 1.0) < 1e-12,
          f"w = ({w[0]:.4f}, {w[1]:.4f})")

    print("\n5. combined gap G = max(g, -m): the four cases")
    m_t = torch.tensor([3.0, 0.0, 3.0, 0.0], dtype=torch.float64)
    g_t = torch.tensor([-2.0, -2.0, 1.0, 0.0], dtype=torch.float64)
    G = combined_gap(m_t, g_t, "both")
    w = torch.exp(log_weight(G, 0.5))
    labels = ["secure owner, target clear ",
              "secure owner, target vague ",
              "owns nothing, target clear ",
              "on both boundaries         "]
    for lab, gv, wv in zip(labels, G.tolist(), w.tolist()):
        print(f"       {lab}  G = {gv:+5.2f}   w = {wv:.4f}")
    check("secure + clear gets the least", w[0] == w.min())
    check("owns nothing gets the most", w[2] == w.max())
    check("target-ambiguous owner is capped at w = 1/2",
          abs(float(w[1]) - 0.5) < 1e-12)

    print("\n6. mode reductions")
    m_big = torch.full((5,), 1e6, dtype=torch.float64)
    g_any = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    check("m -> infinity gives the ownership-only gate",
          torch.allclose(combined_gap(m_big, g_any, "both"),
                         combined_gap(m_big, g_any, "ownership")))
    g_small = torch.full((5,), -1e6, dtype=torch.float64)
    m_any = torch.tensor([0.0, 0.3, 0.7, 1.2, 3.0], dtype=torch.float64)
    check("g -> -infinity gives the landmark-only gate",
          torch.allclose(combined_gap(m_any, g_small, "both"),
                         combined_gap(m_any, g_small, "landmark")))

    print("\n7. kappa calibration and the frozen budget")
    pos = rng.uniform(-1.2, 1.2, size=(4000, 3, 2))
    lm = rng.uniform(-1.2, 1.2, size=(4000, 3, 2))
    ref = np.concatenate([synthetic_observations(p, l)
                          for p, l in zip(pos, lm)], axis=0)
    gate = OwnershipGate.fit(ref, alpha=0.01, mode="both", decades=1.0)
    print(f"       {gate}")
    m_r, g_r = gaps(torch.as_tensor(ref))
    G_r = combined_gap(m_r, g_r, "both")
    lo = torch.quantile(G_r.double(), 0.10).item()
    hi = torch.quantile(G_r.double(), 0.90).item()
    closed = (hi - lo) / (2 * LN10)
    print(f"       G_10 = {lo:+.4f}   G_90 = {hi:+.4f}   "
          f"closed-form kappa = {closed:.4f}")
    d = gate.diagnostics(torch.as_tensor(ref))
    check("alpha_eff spans exactly one decade across the bulk",
          abs(d["interdecile_log_alpha"] - LN10) < 1e-6,
          f"interdecile log alpha_eff = {d['interdecile_log_alpha']:.8f} "
          f"(ln 10 = {LN10:.8f})")
    check("geometric mean of alpha_eff equals alpha on the reference set",
          abs(d["alpha_eff_geomean"] / 0.01 - 1.0) < 1e-9,
          f"geomean = {d['alpha_eff_geomean']:.10f}")
    check("arithmetic mean is ABOVE alpha, as Jensen requires",
          d["alpha_eff_mean"] > d["alpha_eff_geomean"],
          f"arith = {d['alpha_eff_mean']:.6f} > "
          f"geo = {d['alpha_eff_geomean']:.6f}")
    print(f"       alpha_eff q10 = {d['alpha_eff_q10']:.5f}   "
          f"q90 = {d['alpha_eff_q90']:.5f}   "
          f"clamped = {100 * d['frac_clamped']:.2f}%   "
          f"owns nothing = {100 * d['frac_owns_nothing']:.1f}%")

    print("\n8. predicted sigma field against the actor's clamps")
    s_lo = 0.656 * math.sqrt(d["alpha_eff_q10"] / 0.01)
    s_hi = 0.656 * math.sqrt(d["alpha_eff_q90"] / 0.01)
    s_min = 0.656 * math.sqrt(1.0 / gate.clamp_ratio)
    s_max = 0.656 * math.sqrt(gate.clamp_ratio)
    print(f"       bulk  sigma* in [{s_lo:.4f}, {s_hi:.4f}]")
    print(f"       clamp sigma* in [{s_min:.4f}, {s_max:.4f}]")
    check("whole clamped range sits inside LOG_STD bounds "
          "[0.0067, 7.39]", 0.0067 < s_min and s_max < 7.389)

    print("\n9. reduction to A1 as kappa -> infinity")
    huge = OwnershipGate(alpha=0.01, kappa=1e9,
                         log_z=fit_log_z(G_r, 1e9), mode="both")
    a = huge(torch.as_tensor(ref))
    check("alpha_eff -> alpha", float((a / 0.01 - 1.0).abs().max()) < 1e-8,
          f"max rel dev = {float((a / 0.01 - 1.0).abs().max()):.2e}")
    print("     -> w -> 1/2 everywhere and Z -> 1/2, so the ratio is 1. "
          "Bit-identity for R1 still uses alpha_fn=None.")

    print("\n10. stop-grad and shapes")
    o = torch.as_tensor(ref[:768].reshape(256, 3, -1),
                        dtype=torch.float32).requires_grad_(True)
    a = gate(o)
    check("output shape is [B, N]", tuple(a.shape) == (256, 3),
          f"{tuple(a.shape)}")
    check("output carries no gradient", not a.requires_grad)
    check("output is finite and positive",
          bool(torch.isfinite(a).all() and (a > 0).all()))

    print("\n11. m agrees with the Experiment-3 token")
    try:
        from token_diversity import tokenise
        pos = rng.uniform(-1.2, 1.2, size=(3, 2))
        lm = rng.uniform(-1.2, 1.2, size=(3, 2))
        o = synthetic_observations(pos, lm)
        tok = tokenise(o)
        L = o[:, LANDMARK_BLOCK].reshape(3, N_LANDMARKS, 2)
        own = np.argmin(np.linalg.norm(L, axis=-1), axis=-1)
        m_o, _ = gaps(torch.as_tensor(o))
        check("nearest-landmark index matches token_diversity",
              bool(np.array_equal(tok, own)))
        check("m >= 0 always", bool((m_o >= 0).all()))
    except Exception as exc:
        check("token_diversity cross-check", False, f"{exc}")

    print("\n12. kill switch fires on a near-constant gap")
    try:
        fit_kappa(np.full(1000, 0.7))
        check("raises on degenerate reference set", False)
    except ValueError:
        check("raises on degenerate reference set", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
