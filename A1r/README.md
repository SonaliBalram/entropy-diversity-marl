# A1r — ownership-gated entropy allocation

Everything for Experiment A1r lives in this folder. Experiments 1, 2 and 3
are untouched: they keep using `../maddpg_system.py`, and nothing here is
imported by them.

A1r spends **the same entropy budget as A1** (α = 0.01) but distributes it
unevenly across states:

```
alpha_eff,i = alpha * w_i / Z        w_i = 1 / (1 + exp(-G_i / kappa))
Z = exp( E_reference[log w] )        the GEOMETRIC mean, frozen
```

The geometric mean is the correct normaliser: the budget being matched is
mean realised entropy, and since σ\* ∝ √α_eff we have H ∝ ½ log α_eff, so
holding mean entropy fixed means holding E[log α_eff] fixed. Normalising by
E[w] instead would silently underspend by log E[w] − E[log w] nats.

## The gate

Two Voronoi diagrams, combined into one signed gap:

```
G_i = max( g_i , -m_i )

g_i = min_k [ rho_ik - min_(j!=i) rho_jk ]    sites = AGENTS   (signed)
m_i = rho_i(2) - rho_i(1)                     sites = LANDMARKS (>= 0)
```

`g_i = 0` is exactly the perpendicular bisector between agent *i* and its
nearest rival for landmark *k*, so ownership flips exactly at the boundary
and the logistic gate sits at w = ½ there.

**Why `g` is the right signal.** The team reward is `R = -sum_k min_j rho_jk`,
and that `min_j` *is* the agent-sited diagram. Agent *i*'s marginal
contribution is `Delta_i = sum_k max(0, -c_ik)`, so `Delta_i = 0` exactly
when `g_i >= 0`. An agent with `g_i >= 0` contributes exactly nothing to the
reward, so noise there is free. That also puts `g` in reward units, the same
units as `m`, which is what lets one κ serve both.

Because m ≥ 0, the landmark term can only lift the gate as far as w = ½,
while dispossession can carry it to w → 1. Deliberate: dispossession is the
stronger statement.

## Files

| file | what it is |
|---|---|
| `voronoi_gate.py` | the geometry, the gate, κ and Z fitting. `python A1r/voronoi_gate.py` runs 26 self-tests. |
| `a1r_pilot.py` | **run first.** Kill switch, freezes κ and Z, τ baseline. |
| `maddpg_system_a1r.py` | copy of `../maddpg_system.py` + the `alpha_fn` hook. Header records the parent's SHA256 and lists every change. |
| `train_a1r.py` | copy of `../exp2/train_exp2.py` reading the frozen constants. |
| `test_reductions_a1r.py` | R1–R5. |
| `a1r_tau.py` | the tracking coefficient. |

## Order of work

```bash
python A1r/voronoi_gate.py                     # 1. self-test
python A1r/a1r_pilot.py                        # 2. kill switch + freeze
python A1r/test_reductions_a1r.py              # 3. R1-R5
python A1r/train_a1r.py --alpha 0.01 --seed 42 \
    --batch-size 256 --update-every 2 --checkpoint-every 2000
python A1r/a1r_tau.py --ckpt-root checkpoints/a1r_both   # 5. tau FIRST
python exp3_outcome.py ...                     # 6. then Delta D
```

Results → `results/a1r_<gate>/seed_<S>/`, weights →
`checkpoints/a1r_<gate>/seed_<S>/`. Nothing overwrites an existing arm.

## Why a separate copy of maddpg_system.py is safe

`--gate none` sets `alpha_fn=None`, and the two original actor-loss branches
in the copy are byte-identical to the parent's. R1 runs
`exp2/train_exp2.py --alpha 0.01` against `train_a1r.py --gate none` and
requires every logged array to be bit-identical. That is an empirical
integrity check on the copy, stronger than comparing hashes. It is also
stronger than anything Exp 3 could manage: A1r holds no transitions, so it
cannot desynchronise the replay buffer the way the deferred-write design did.

## Two things to keep reporting

**The realised budget drifts.** Z is frozen so the gate stays a fixed
function of the observation. The price is that the realised geometric mean
of α_eff moves as training moves the state distribution — measured on the A1
ladder as 1.58× α at episode 2,000 settling to 0.85× by 50,000 (σ factors
1.26 → 0.92, inside A1's own ±10% seed spread). `train_a1r.py` logs
`gate_geomeans` every update so this is reported, not hidden. Correcting it
online would reintroduce the moving scale the freeze exists to prevent.

**The allocation can flatten.** `gate_spreads` logs the interdecile spread of
log α_eff in decades. It is 1.0 by construction on the reference set. If it
collapses towards 0 during training, A1r has quietly become A1 and the
comparison is vacuous.

## Pilot result (22 Sep 2026, 360,000 reference states)

```
verdict            RUN A1r AS DESIGNED
kappa              0.349547
log Z              -1.531728
interdecile G      1.0777 reward units
owns nothing       14.28% of samples
diagram binding    43.95% landmark-sited / 56.05% agent-sited
corr(g, -m)        +0.6854   -- correlated, but m has ZERO lift for
                                predicting ownership (85.7% = base rate),
                                so the agent-sited term is not redundant
tau baseline       +0.0282 (r = 0.081) -- A1 does not already track the gate
```
