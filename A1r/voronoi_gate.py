"""
voronoi_gate.py

Experiment A1r -- the ownership gate.

WHAT THIS IS
------------
A1r spends the SAME total entropy budget as A1 (alpha = 0.01) but
distributes it unevenly across states.  The coefficient that multiplies
the policy entropy in the actor loss becomes a function of the agent's
own observation:

    alpha_eff,i = alpha * w_i / Z,        w_i = 1 / (1 + exp(-G_i / kappa))

    Z = exp( E_reference[ log w ] )        (the GEOMETRIC mean of w)

The geometric mean is the right normaliser, not the arithmetic one.  The
budget being matched is the mean realised entropy, and

    sigma* propto sqrt(alpha_eff)  =>  H = log sigma + const
                                    =>  H propto (1/2) log alpha_eff

so holding the mean entropy fixed means holding E[log alpha_eff] fixed,
which needs E[log w] = log Z.  Normalising by E[w] instead would silently
underspend the budget by log E[w] - E[log w] nats (Jensen).

TWO VORONOI DIAGRAMS
--------------------
1.  Sites = LANDMARKS.  Agent i sits in the cell of its nearest landmark.
    Distance to that boundary, in decision terms:

        m_i = rho_i(2) - rho_i(1)                >= 0

    m_i = 0 is exactly where argmin_k rho_ik flips, i.e. where the
    Experiment-3 token changes letter.  Gate and outcome measure share
    one geometric object.

2.  Sites = AGENTS.  Landmark k falls in the cell of whichever agent is
    nearest to it; the boundary between agents i and j is the
    perpendicular bisector of the segment joining them, i.e. the locus
    rho_ik = rho_jk.  The signed distance-difference from that bisector is

        c_ik = rho_ik - min_{j != i} rho_jk
        g_i  = min_k c_ik                        (signed)

    g_i <  0  ->  agent i is the nearest agent to at least one landmark
    g_i =  0  ->  agent i is exactly on the bisector for its best landmark
    g_i >  0  ->  every landmark has a closer rival; agent i owns nothing

    Ownership therefore changes EXACTLY at the boundary, and the logistic
    gate's half-point w = 1/2 sits exactly on it.

WHY g IS THE RIGHT SIGNAL (the reward-units argument)
-----------------------------------------------------
Cooperative Navigation's team reward is  R = -sum_k min_j rho_jk  (minus
collisions).  That min_j IS the agent-sited Voronoi diagram.  Agent i's
marginal contribution -- what the team loses if i vanishes -- is

    Delta_i = sum_k [ min_{j!=i} rho_jk - min_j rho_jk ]
            = sum_k max(0, -c_ik)                             (exact)

because min_j rho_jk = min(rho_ik, min_{j!=i} rho_jk).  Hence

    Delta_i = 0   <=>   g_i >= 0                              (exact)

and generically (agent i nearest to at most one landmark) Delta_i =
max(0, -g_i).  So -g_i IS agent i's marginal contribution to the team
reward, measured in reward units.  g_i >= 0 means its contribution is
exactly zero -- its actions are locally reward-neutral, so noise there
costs the team nothing.  That is why dispossessed agents get the entropy.
It also puts g in the same units as m, which is what lets one kappa serve
both.

COMBINING THE TWO
-----------------
    G_i = max( g_i , -m_i )

Since m_i >= 0, the landmark term can only raise the gate as far as
w = 1/2 (boundary level), while dispossession can carry it to w -> 1.
Read it as:

    owns nothing                      ->  full entropy      (w -> 1)
    owns something, target ambiguous  ->  boundary entropy  (w  = 1/2)
    secure owner, target clear        ->  almost none       (w -> 0)

The asymmetry is deliberate: dispossession is the stronger statement
(marginal contribution exactly zero) than ambiguity (both targets equally
good, but the agent still contributes).  Setting m -> infinity recovers
g alone, so the ownership-only arm nests inside this one.

FROZEN kappa AND Z
------------------
Both are estimated ONCE, before training, from a reference set of states
(see a1r_pilot.py), and never re-estimated.  Training changes the state
distribution, so a Z re-estimated online would make alpha_eff drift for
reasons that have nothing to do with the state, and the intervention
would stop being "state-dependent entropy" and become "state-dependent
entropy whose scale moves because the policy moved".  The realised
geometric mean is instead LOGGED at every checkpoint, so that drift is
measured and reported rather than absorbed into the design.

WHERE IT ENTERS
---------------
The actor loss ONLY, as a per-(sample, agent) coefficient of shape
[B, N], under stop-grad:

    L_actor = -E[Q] - E[ stopgrad(alpha_eff,i) * H(pi_i) ]

Not the reward, not the critic target, no buffer change (the gaps are
recomputed from the observation batch), current observations not next,
and the same entropy estimator A1 uses.

Note on the stop-grad: alpha_eff depends only on observations, so no
gradient reaches it in any case; the detach makes that explicit and
stops an accidental graph being built.  It blocks the DIRECT path -- the
policy cannot differentiate through its own temperature.  It does not
block the indirect path, since the policy still chooses which states it
visits.  What closes that hole is the normalisation, not the stop-grad:
drifting towards states where w is uniformly high raises Z by the same
factor, so the arm can reallocate entropy but cannot inflate the total.

Run  `python voronoi_gate.py`  for the self-test.
"""

import math

# --------------------------------------------------------------------
# This file lives in A1r/.  The shared algorithm modules (environment.py,
# token_diversity.py, models/, ...) live one level up, so put that folder
# on the import path and anchor output directories to it -- results land
# in the SHARED results/ tree whichever directory the script is run from.
# --------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Observation layout: simple_spread_v3, N = 3, observation_dim = 18.
# Mirrors environment.py lines 38-43 and token_diversity.py's
# LANDMARK_BLOCK.  Changing n_agents changes these.
# ---------------------------------------------------------------------
LANDMARK_BLOCK    = slice(4, 10)    # 3 x 2, entries are  l_k - x_i
OTHER_AGENT_BLOCK = slice(10, 14)   # 2 x 2, entries are  x_j - x_i
N_LANDMARKS = 3
N_RIVALS    = 2

LN10 = math.log(10.0)

MODES = ("both", "ownership", "landmark")


# =====================================================================
# 1.  GEOMETRY
# =====================================================================

def gaps(obs):
    """
    The two margins, straight from each agent's own observation.

    obs : [..., observation_dim].  Training passes [B, N, obs_dim];
          the pilot passes [n_samples, obs_dim].  Any leading shape works.

    Returns (m, g), each of shape obs.shape[:-1].

    The rival distances are EXACT, not approximated.  Agent i observes
        L_k = l_k - x_i      (landmark block)
        A_j = x_j - x_i      (other-agent block)
    so
        l_k - x_j = (l_k - x_i) - (x_j - x_i) = L_k - A_j
    and rho_jk = || L_k - A_j ||.  Everything the gate needs is therefore
    a function of o_i alone, which is what makes alpha_eff representable
    by the state-conditioned sigma head -- the premise the tracking
    coefficient tau is testing.
    """
    o = torch.as_tensor(obs)
    lead = o.shape[:-1]

    L = o[..., LANDMARK_BLOCK].reshape(*lead, N_LANDMARKS, 2)
    A = o[..., OTHER_AGENT_BLOCK].reshape(*lead, N_RIVALS, 2)

    # own distance to each landmark                          [..., K]
    rho_self = torch.linalg.vector_norm(L, dim=-1)

    # every rival's distance to every landmark               [..., K, J]
    rho_rival = torch.linalg.vector_norm(
        L.unsqueeze(-2) - A.unsqueeze(-3), dim=-1
    )

    # closest rival for each landmark, then the ownership gap
    c = rho_self - rho_rival.min(dim=-1).values              # [..., K]
    g = c.min(dim=-1).values                                 # [...]

    # landmark margin: second-nearest minus nearest
    top2 = torch.topk(rho_self, 2, dim=-1, largest=False).values
    m = top2[..., 1] - top2[..., 0]                          # [...]

    return m, g


def combined_gap(m, g, mode="both"):
    """
    G_i, the signed quantity the logistic gate acts on.

    "both"      G = max(g, -m)   both diagrams          <- A1r
    "ownership" G = g            agent-sited only
    "landmark"  G = -m           landmark-sited only
    """
    if mode == "both":
        return torch.maximum(g, -m)
    if mode == "ownership":
        return g
    if mode == "landmark":
        return -m
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def log_weight(G, kappa):
    """
    log w = log sigmoid(G / kappa), computed with logsigmoid.

    log(sigmoid(x)) underflows to -inf for x below about -750, which is
    reachable for a very secure owner at small kappa.  logsigmoid is
    stable there.  Working in log space throughout is also exactly how
    the budget identity is derived, so nothing is exponentiated until
    the last line.
    """
    return F.logsigmoid(torch.as_tensor(G) / kappa)


# =====================================================================
# 2.  FREEZING kappa AND Z FROM A REFERENCE SET
# =====================================================================

def fit_kappa(G_ref, decades=1.0, q_lo=0.10, q_hi=0.90):
    """
    Choose kappa so that alpha_eff spans exactly `decades` decades
    between the q_lo and q_hi quantiles of the reference gaps.

    Because logsigmoid is increasing, the quantiles of log w are the
    logsigmoids of the quantiles of G, so the interdecile spread of
    log alpha_eff is the scalar

        s(kappa) = logsigmoid(G_hi / kappa) - logsigmoid(G_lo / kappa)

    s is strictly decreasing in kappa (s -> infinity as kappa -> 0,
    s -> 0 as kappa -> infinity), so a bisection on log kappa is exact
    to machine precision.  Reported alongside is the closed-form
    approximation (G_hi - G_lo) / (2 ln 10), which is what you get if
    the gap distribution happens to be centred on the boundary.

    KILL SWITCH.  If the reference gaps are near-constant there is no
    allocation to make and the arm should not be run; this raises
    rather than returning a meaningless kappa.
    """
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
    """
    log Z = E_reference[ log w ].  Frozen; never re-estimated online.
    """
    g = torch.as_tensor(np.asarray(G_ref, dtype=np.float64)).flatten()
    return float(log_weight(g, kappa).mean())


# =====================================================================
# 3.  THE GATE
# =====================================================================

class OwnershipGate:
    """
    Callable turning an observation batch into alpha_eff of shape [B, N].

    Constructed with kappa and log_z already frozen.  Pass an instance
    to MADDPGSystem(alpha_fn=...); it is called inside the actor update
    and its output is detached.
    """

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

    # -- the coefficient -------------------------------------------
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

    # -- diagnostics logged at every checkpoint ---------------------
    def diagnostics(self, obs):
        """
        Everything needed to see whether the frozen Z is still honest
        and whether the allocation still has any spread left in it.

        realised_geometric_mean  should equal alpha if the visited state
                                 distribution still matches the reference
                                 set.  Drift away from alpha is the price
                                 of freezing Z, and it is reported, not
                                 hidden.
        interdecile_log_alpha    ln 10 by construction on the reference
                                 set.  If this collapses towards 0 the
                                 allocation has flattened and A1r has
                                 quietly become A1.
        """
        with torch.no_grad():
            m, g = gaps(obs)
            G = combined_gap(m, g, self.mode)
            log_a = self._log_alpha + log_weight(G, self.kappa) - self.log_z
            flat = log_a.flatten().double()
            q10 = torch.quantile(flat, 0.10).item()
            q90 = torch.quantile(flat, 0.90).item()
            return {
                # which of the two Voronoi diagrams is setting G.
                # G = max(g, -m), so the landmark-sited diagram decides
                # exactly when -m > g.  If this sits near 0 or near 1 the
                # arm is effectively using only one of the two diagrams.
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

    # -- persistence ------------------------------------------------
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
        """Freeze kappa and log_z from a reference observation set."""
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


# =====================================================================
# 4.  TEST HELPERS
# =====================================================================

def synthetic_observations(positions, landmarks):
    """
    Build observations exactly as simple_spread_v3 lays them out, so the
    self-test can compare the gate against ground truth it controls.

    positions : [N, 2]      landmarks : [K, 2]

    Other agents appear in world order with self skipped, which is what
    MPE does.  The ordering is irrelevant to the gate (it takes a min
    over rivals) but the test should match the real thing.
    """
    positions = np.asarray(positions, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    n = positions.shape[0]
    dim = 4 + 2 * landmarks.shape[0] + 2 * (n - 1) + 2 * (n - 1)
    obs = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        obs[i, 0:2] = 0.0                     # velocity: unused here
        obs[i, 2:4] = positions[i]
        obs[i, LANDMARK_BLOCK] = (landmarks - positions[i]).reshape(-1)
        others = [j for j in range(n) if j != i]
        obs[i, OTHER_AGENT_BLOCK] = (
            positions[others] - positions[i]).reshape(-1)
    return obs


def brute_force(positions, landmarks):
    """
    m, g and the marginal contribution, computed from global positions
    with plain loops.  Deliberately slow and obvious.
    """
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


# =====================================================================
# 5.  SELF-TEST
# =====================================================================

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
    # two agents fixed, third parked far away; a landmark walks across
    # the bisector of agents 0 and 1
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
    # two agents contesting one landmark (all three landmarks coincide),
    # third agent far away
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
    # sigma* propto sqrt(alpha_eff), pinned to A1's measured 0.656 at alpha
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
    except Exception as exc:                      # noqa: BLE001
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
