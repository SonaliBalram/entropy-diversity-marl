# Entropy-Induced Diversity in Cooperative MARL

Honours research project, Computational and Applied Mathematics, University of the Witwatersrand (2026).

**Question:** can entropy regularisation make agents that share one policy behave differently from each other, and at which level of the learning process (actions, trajectories, roles) does that diversity appear?

**Setup:** MADDPG under centralised training with decentralised execution (CTDE). Three agents share one stochastic actor, and there are twin centralised critics (clipped double-Q). The task is MPE Cooperative Navigation (`simple_spread_v3`), trained for 50,000 episodes with four seeds (42, 128, 456, 789).

## Experiment arms

| Arm | What changes | Training script | Results folder |
|---|---|---|---|
| A0 | Baseline, no entropy bonus (alpha = 0) | `train_baseline.py` | `results/baseline/` |
| A1 | Policy-entropy bonus, alpha = 0.01 | `exp2/train_exp2.py --alpha 0.01` | `results/alpha_0.01/` |
| A1 (0.05) | Policy-entropy bonus, alpha = 0.05 | `exp2/train_exp2.py --alpha 0.05` | `results/alpha_0.05/` |
| A1r | Entropy budget allocated by a Voronoi ownership gate | `A1r/run_a1r.sh` | `results/a1r_both/` |
| A2 | Trajectory-diversity bonus on token D, beta = 3 | `exp3/train_exp3.py --beta 3` | `results/beta_3/` |
| A2c | A2 with the augmented critic | `exp3/train_exp3.py --beta 3 --augment-critic` | `results/beta_3_augcritic/` |
| A2z | Capacity control: augmented critic, no bonus (beta = 0) | `run_a2z.sh` | `results/beta_0_augcritic/` |
| A2w | Windowed trajectory-diversity bonus, beta = 3, 5 windows per episode | `exp3_windowed/train_a2w.py --beta 3` | `results/beta_3_w5/` |
| A2cw | A2w with the critic given the windowed context | `exp3_windowed/train_a2w.py --beta 3 --augment-critic` | `results/beta_3_w5_augcritic/` |
| A2cwT | A2cw with the final step treated as terminal | `exp3_windowed/train_a2w.py --beta 3 --augment-critic --terminal-at-limit` | `results/beta_3_w5_augcritic_term/` |

Log files use older names for the Exp 3 arms: `exp3` = A2, `exp3c` = A2c, `exp3z` = A2z.

Run any script with `--help` to see all of its options.

## Folder layout

| Path | Contents |
|---|---|
| `environment.py`, `maddpg_system.py`, `replay_buffer.py`, `action_utils.py`, `models/` | Core algorithm: environment wrapper, MADDPG system, replay buffer, actor and critic networks |
| `train_baseline.py`, `exp2/`, `exp3/`, `A1r/`, `run_a2z.sh` | Training, one entry point per experiment |
| `exp3_windowed/` | Windowed arms (A2w, A2cw, A2cwT): training (`train_a2w.py`), the windowed diversity measure (`windowed_diversity.py`), and their analysis (`block4_checks.py`, `analyse_arms.py`, results in `results/exp3_windowed/`) |
| `instantaneous_metrics.py`, `temporal_behaviour.py`, `policy_divergence_behaviour.py`, `token_diversity.py` | Behavioural diversity measures |
| `evaluate_baseline.py`, `evaluate_checkpoints.py`, `token_eval.py`, `exp3_*.py`, `analyse_run.py` | Held-out evaluation and analysis |
| `propagation_matrix.py` | Matched-return comparison of each arm against its control at the action, trajectory and arena levels (`results/propagation_matrix/`) |
| `team_coverage.py` | Exploratory team-coverage measure on a 3x3 arena grid, reusing the propagation-matrix arms (`results/team_coverage/`) |
| `make_report_figures.py` | Figures for the report (`results/report_figures/`); needs the per-episode training histories, which are not in this repository |
| `test_reductions.py`, `A1r/test_reductions_a1r.py`, `exp3_windowed/test_a2w.py`, `exp3_windowed/test_windowed_diversity.py` | Correctness checks: each new arm must reduce exactly to an earlier arm when its new parameter is switched off |
| `results/` | Evaluation and analysis outputs (per-episode training histories are not included, see below) |
| `checkpoints/` | Only the `best.pt` model for each arm and seed |
| `docs/` | Supporting notes |

## Installation

Developed with Python 3.9.6.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Example training run (Experiment 2, seed 42):

```bash
.venv/bin/python exp2/train_exp2.py --alpha 0.01 --seed 42 --batch-size 256 --update-every 2 --checkpoint-every 2000
```

## Checking the code

```bash
.venv/bin/python test_reductions.py
.venv/bin/python A1r/test_reductions_a1r.py
.venv/bin/python exp3_windowed/test_a2w.py
.venv/bin/python exp3_windowed/test_windowed_diversity.py
```

These scripts run short training runs in a throwaway copy of the project and never touch `results/` or `checkpoints/`.

## Not included in this repository

These are kept offline because of their size (about 2.3 GB in total):

- per-episode training histories (`results/<arm>/seed_*/`)
- intermediate checkpoints (`checkpoints/<arm>/seed_*/ckpt_ep*.pt`)
- full training logs (`logs/`)
