# autoresearch — Tribal Village

This is an autonomous MARL research loop. The agent modifies `train.py`, runs experiments with a fixed time budget, measures representation geometry (EffRank/n), and keeps improvements.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — environment context.
   - `train.py` — the file you modify. MAPPO training loop, reward structure, contrastive loss.
   - `program.md` — this file.
4. **Verify the environment builds**: run `python train.py --dry-run` to confirm the library loads.
5. **Initialize results.tsv**: create it with just the header row.
6. **Confirm and go**.

## What you are optimizing

The research goal is: **learn one geometrically distinct representation per agent role**.

The primary metric is `effrank_n` (EffRank divided by number of agents), measured at a fixed training checkpoint. Higher is better: `effrank_n > 1` means the shared encoder maintains more independent directions than agents.

The secondary metric is `probe_acc` (linear probe accuracy for role prediction from frozen encoder). Chance is 0.50; higher is better.

The ground truth comparison:
- **Individual rewards** should produce `effrank_n > 1` and high `probe_acc`.
- **Shared rewards** should collapse both to near 0.5 probe accuracy and `effrank_n < 1`.

Your job is to find configurations that maximize `effrank_n` under the shared-reward condition (the hard case), or to validate the causal story cleanly.

## What you CAN do (modify `train.py`)

- Reward structure: individual vs. shared vs. mixed (e.g. `0.8 * team + 0.2 * individual`)
- Contrastive loss weight (`alpha_c`), temperature, and whether it's on or off
- MAPPO hyperparameters: learning rate, entropy coefficient, PPO clip
- Number of agents (start with 12; can try 6 or 24)
- Encoder architecture: depth, width, normalization
- Advantage normalization: per-agent vs. global

## What you CANNOT do

- Modify `prepare_env.py` (the evaluation harness and environment setup are fixed).
- Add new Python packages. Use only what is already installed in the `tribal-village` conda env.
- Change the EffRank metric computation — it must remain the SVD-based formula in `train.py`.

## Running an experiment

```bash
conda run -n tribal-village python train.py > run.log 2>&1
```

The script runs for a **fixed 10-minute wall-clock budget** (excluding startup), then prints a summary:

```
---
effrank_n:        1.234
probe_acc:        0.712
d_act:            0.089
training_seconds: 600.1
total_seconds:    623.4
num_steps:        48000
num_agents:       12
reward_type:      individual
```

Extract the key metrics:
```bash
grep "^effrank_n:\|^probe_acc:\|^reward_type:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated). Header and columns:

```
commit	effrank_n	probe_acc	reward_type	status	description
```

- `commit`: short git hash (7 chars)
- `effrank_n`: EffRank/n at the end of training (0.000 for crashes)
- `probe_acc`: linear probe accuracy (0.000 for crashes)
- `reward_type`: `individual`, `shared`, or `mixed`
- `status`: `keep`, `discard`, or `crash`
- `description`: short text description of the experiment

Example:
```
commit	effrank_n	probe_acc	reward_type	status	description
a1b2c3d	1.234	0.712	individual	keep	baseline individual rewards 12 agents
b2c3d4e	0.381	0.503	shared	keep	baseline shared rewards 12 agents (collapse confirmed)
c3d4e5f	0.890	0.611	mixed	keep	0.2 individual + 0.8 shared partially recovers
d4e5f6g	1.310	0.741	individual	keep	individual + contrastive alpha=6.8e-4 sharpens roles
e5f6g7h	0.290	0.501	shared	discard	shared + contrastive makes it worse as expected
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch and last commit).
2. Modify `train.py` with one experimental idea.
3. `git commit -am "experiment: <short description>"`
4. `conda run -n tribal-village python train.py > run.log 2>&1`
5. `grep "^effrank_n:\|^probe_acc:\|^reward_type:\|^training_seconds:" run.log`
6. If grep is empty: run crashed. `tail -n 50 run.log` to diagnose. Fix if obvious; otherwise discard and move on.
7. Record in `results.tsv` (do not commit this file).
8. If `effrank_n` improved (higher) or the experiment answered a causal question cleanly: advance the branch (keep the commit).
9. Otherwise: `git reset --hard HEAD~1` and revert to the previous state.

**Priority order for experiments:**
1. Establish baselines: individual rewards (should show high effrank_n) and shared rewards (should show collapse). These are the most important data points for the paper.
2. Mixed rewards: find the threshold where collapse begins.
3. Contrastive loss: does it help under individual rewards? Does it hurt under shared rewards?
4. Architecture changes that affect EffRank: encoder depth, layer normalization, spectral normalization.

**Timeout**: if a run exceeds 15 minutes, kill it. Treat as crash.

**NEVER STOP**: run autonomously until manually interrupted. Do not ask for confirmation between experiments.
