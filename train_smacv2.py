"""
SMACv2 representation geometry training script.
Mirrors train.py but uses the SMACv2 centralized-step API.

Unit types: marine=role 0, marauder=role 1, medivac=role 2.
Individual rewards use the SC2 API directly:
  - Attackers (marine/marauder): delta-HP of their targeted enemy + kill bonus,
    split equally among agents that targeted the same enemy in the same step.
  - Healers (medivac): delta-HP restored to their targeted ally.
  - Move/stop/dead agents: 0 (plus a death penalty for agents that just died).
  All values are scaled by the same factor as SMAC's team reward.

Usage:
  python train_smacv2.py --reward-type individual
  python train_smacv2.py --reward-type shared
  python train_smacv2.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

warnings.filterwarnings("ignore")
os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# =============================================================================
# CONFIG
# =============================================================================

REWARD_TYPE          = "shared"
NUM_UNITS            = 6          # allied agents per team
NUM_ROLES            = 3          # marine / marauder / medivac
LEARNING_RATE        = 3e-4
ENTROPY_COEF         = 0.01
PPO_CLIP             = 0.2
VF_COEF              = 0.5
GAMMA                = 0.99
GAE_LAMBDA           = 0.95
UPDATE_EPOCHS        = 4
MINIBATCH_SIZE       = 256
ROLLOUT_STEPS        = 128        # smaller than tribal-village (SC2 steps are heavier)
ENCODER_HIDDEN       = 256
ENCODER_LAYERS       = 2
TRAIN_BUDGET_SECONDS = 1800       # 30-min hard cap; convergence check stops earlier
PROBE_EVAL_EPISODES  = 5          # collect this many episodes for probe
SEED                 = 0

# Convergence: EMA-smoothed loss must have std < CONV_STD_TOL over CONV_WINDOW updates.
# EMA filters per-minibatch spikes; std of the smoothed signal detects plateau.
CONV_EMA_ALPHA = 0.1   # smoothing factor (lower = smoother)
CONV_WINDOW    = 30    # updates in rolling window of the EMA loss
CONV_STD_TOL   = 0.03  # std of EMA loss over window → convergence
MIN_UPDATES    = 80    # minimum updates before convergence can trigger (~3 min)

# Where to append a result line (TSV)
RESULTS_FILE = "/Users/tasha/Documents/Github/tribal-village/results_smacv2.tsv"

MAP_NAME = "10gen_terran"


# =============================================================================
# Environment
# =============================================================================

def make_env(n_units: int, seed: int = 0):
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    distribution_config = {
        "n_units": n_units,
        "n_enemies": n_units,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": [],
            "weights": [1/3, 1/3, 1/3],   # equal distribution → roughly 2 of each role
            "observe": True,               # unit type is in obs
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": n_units,
            "map_x": 32,
            "map_y": 32,
        },
    }
    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name=MAP_NAME,
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        step_mul=8,
        seed=seed,
    )
    return env


def get_role_labels(env) -> np.ndarray:
    """Return role label for each agent based on current unit type assignment."""
    inner = env.env  # StarCraft2Env
    marine_id   = inner.marine_id
    marauder_id = inner.marauder_id
    medivac_id  = inner.medivac_id
    role_map = {marine_id: 0, marauder_id: 1, medivac_id: 2}
    n = env.n_agents
    labels = np.array([role_map.get(inner.agents[i].unit_type, 0) for i in range(n)],
                      dtype=np.int64)
    return labels


def get_obs_array(env) -> np.ndarray:
    """Return obs as (n_agents, obs_dim) float32 array."""
    obs_list = env.get_obs()
    return np.stack(obs_list).astype(np.float32)


def get_individual_rewards(env, actions: list[int]) -> np.ndarray:
    """
    Per-agent individual rewards from SC2 API health deltas.

    Attackers credit: damage dealt to their targeted enemy + kill bonus,
    split equally among co-attackers of the same target.
    Medivac credit: HP restored to targeted ally.
    Dying agent penalty: -reward_death_value * neg_scale.
    All values use SMAC's reward_scale factor.
    """
    inner = env.env  # StarCraft2Env
    n = env.n_agents
    rewards = np.zeros(n, dtype=np.float32)

    neg_scale = inner.reward_negative_scale
    n_no_attack = inner.n_actions_no_attack
    is_terran = inner.map_type in ["MMM", "terran_gen"]

    # Map target → list of agent ids for splitting credit
    enemy_attackers: dict[int, list[int]] = {}
    ally_healers:    dict[int, list[int]] = {}

    for a_id, action in enumerate(actions):
        unit = inner.agents[a_id]
        if unit.health == 0 or action < n_no_attack:
            continue
        target_id = action - n_no_attack
        if is_terran and unit.unit_type == inner.medivac_id:
            ally_healers.setdefault(target_id, []).append(a_id)
        else:
            enemy_attackers.setdefault(target_id, []).append(a_id)

    # Attacker credit: enemy HP delta + kill bonus, split among co-attackers
    for e_id, attacker_ids in enemy_attackers.items():
        prev = inner.previous_enemy_units.get(e_id)
        curr = inner.enemies.get(e_id)
        if prev is None or curr is None:
            continue
        prev_hp = prev.health + prev.shield
        curr_hp  = (curr.health + curr.shield) if curr.health > 0 else 0.0
        damage = max(0.0, prev_hp - curr_hp)
        kill_bonus = inner.reward_death_value if (curr.health == 0 and prev.health > 0) else 0.0
        credit = (damage + kill_bonus) / len(attacker_ids)
        for a_id in attacker_ids:
            rewards[a_id] += credit

    # Medivac credit: ally HP restored
    for al_id, healer_ids in ally_healers.items():
        prev = inner.previous_ally_units.get(al_id)
        curr = inner.agents.get(al_id)
        if prev is None or curr is None:
            continue
        heal = max(0.0, curr.health - prev.health)
        credit = heal / len(healer_ids)
        for a_id in healer_ids:
            rewards[a_id] += credit

    # Death penalty for agents that just died this step
    for a_id in range(n):
        prev = inner.previous_ally_units.get(a_id)
        curr = inner.agents.get(a_id)
        if prev is not None and curr is not None and prev.health > 0 and curr.health == 0:
            rewards[a_id] -= inner.reward_death_value * neg_scale

    # Apply same scaling as SMAC team reward
    if inner.reward_scale and inner.max_reward > 0:
        scale = inner.max_reward / inner.reward_scale_rate
        rewards /= scale

    return rewards


# =============================================================================
# Model
# =============================================================================

class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: int = ENCODER_HIDDEN, n_layers: int = ENCODER_LAYERS):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.actor   = nn.Linear(hidden, n_actions)
        self.critic  = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        z = self.encoder(obs.float())
        return self.actor(z), self.critic(z).squeeze(-1), z

    def get_action_and_value(self, obs: torch.Tensor, avail: torch.Tensor | None = None,
                              action: torch.Tensor | None = None):
        logits, value, z = self(obs)
        if avail is not None:
            logits = logits - 1e10 * (1 - avail)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, z


# =============================================================================
# Diagnostics
# =============================================================================

def effective_rank(Z: torch.Tensor) -> float:
    if Z.shape[0] < 2:
        return 1.0
    Z_c = Z - Z.mean(0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(Z_c, full_matrices=False)
        S = S[S > 1e-8]
        if S.numel() == 0:
            return 1.0
        p = S / S.sum()
        return float(torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item())
    except Exception:
        return float("nan")


def action_diversity(logits_list: list[torch.Tensor]) -> float:
    if len(logits_list) < 2:
        return 0.0
    probs = [F.softmax(l, dim=-1) for l in logits_list]
    kl_sum, count = 0.0, 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            kl_sum += F.kl_div(probs[i].log(), probs[j], reduction="batchmean").item()
            count  += 1
    return kl_sum / max(count, 1)


def compute_probe_accuracy(model: MAPPOAgent, env, device: torch.device,
                            n_episodes: int = PROBE_EVAL_EPISODES) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model.eval()
    X_all, y_all = [], []

    for _ in range(n_episodes):
        env.reset()
        role_labels = get_role_labels(env)
        terminated = False
        while not terminated:
            obs_arr = get_obs_array(env)
            obs_t   = torch.tensor(obs_arr, device=device)
            with torch.no_grad():
                z = model.encoder(obs_t)
            for i in range(env.n_agents):
                X_all.append(z[i].cpu().numpy())
                y_all.append(role_labels[i])

            avail_list = [env.get_avail_agent_actions(i) for i in range(env.n_agents)]
            avail_t    = torch.tensor(np.stack(avail_list), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _, _ = model(obs_t)
            logits_masked = logits - 1e10 * (1 - avail_t)
            actions = Categorical(logits=logits_masked).sample().cpu().numpy().tolist()
            _, terminated, _ = env.step(actions)

    X = np.array(X_all)
    y = np.array(y_all)
    if len(np.unique(y)) < 2:
        return 1.0 / NUM_ROLES

    try:
        cv_folds = min(3, min(np.bincount(y)))
        if cv_folds < 2:
            return float("nan")
        clf    = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=cv_folds)
        return float(scores.mean())
    except Exception:
        return float("nan")


# =============================================================================
# GAE
# =============================================================================

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(len(rewards))):
        next_val  = values[t + 1] if t + 1 < len(values) else 0.0
        delta     = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae  = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values[:len(rewards)]


# =============================================================================
# Main
# =============================================================================

def main(dry_run: bool = False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Creating SMACv2 env ({MAP_NAME}, {NUM_UNITS}v{NUM_UNITS})...", flush=True)
    env = make_env(NUM_UNITS, seed=SEED)

    env.reset()
    n_agents   = env.n_agents
    obs_dim    = env.get_obs_size()
    n_actions  = env.get_total_actions()
    role_labels = get_role_labels(env)

    model = MAPPOAgent(obs_dim, n_actions, ENCODER_HIDDEN, ENCODER_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    if dry_run:
        obs_arr = get_obs_array(env)
        print(f"Dry run OK. obs_dim={obs_dim} n_actions={n_actions} "
              f"n_agents={n_agents} device={device} reward={REWARD_TYPE}", flush=True)
        print(f"Sample role labels: {role_labels.tolist()}", flush=True)
        env.close()
        return

    print(f"Starting: agents={n_agents} reward={REWARD_TYPE} obs_dim={obs_dim} "
          f"n_actions={n_actions} device={device}", flush=True)

    train_start = time.time()
    total_steps = 0
    update_count = 0
    last_effrank_n = float("nan")
    last_d_act     = float("nan")
    last_loss      = float("nan")
    ema_loss: float | None = None   # EMA-smoothed loss
    ema_history: list[float] = []   # history of EMA values for convergence check
    converged = False

    # Rollout buffers (flat: T * n_agents)
    obs_buf, act_buf, logp_buf = [], [], []
    val_buf, rew_buf, done_buf = [], [], []

    obs_arr = get_obs_array(env)

    while True:
        elapsed = time.time() - train_start
        if elapsed >= TRAIN_BUDGET_SECONDS:
            break
        if converged:
            break

        # ── Rollout ────────────────────────────────────────────────────────
        model.eval()
        for _ in range(ROLLOUT_STEPS):
            obs_t    = torch.tensor(obs_arr, device=device)
            avail_np = np.stack([env.get_avail_agent_actions(i) for i in range(n_agents)])
            avail_t  = torch.tensor(avail_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                acts, logps, _, vals, _ = model.get_action_and_value(obs_t, avail=avail_t)

            actions = acts.cpu().numpy().tolist()
            team_reward, terminated, _ = env.step(actions)

            # Reward shaping
            if REWARD_TYPE == "shared":
                r = np.full(n_agents, team_reward, dtype=np.float32)
            else:
                # Individual: per-agent HP-delta credit from SC2 API
                r = get_individual_rewards(env, actions)

            obs_buf.append(obs_arr.copy())
            act_buf.append(np.array(actions, dtype=np.int64))
            logp_buf.append(logps.cpu().numpy())
            val_buf.append(vals.cpu().numpy())
            rew_buf.append(r)
            done_buf.append(float(terminated))

            if terminated:
                env.reset()
                role_labels = get_role_labels(env)

            obs_arr = get_obs_array(env)
            total_steps += n_agents

        # ── Update ─────────────────────────────────────────────────────────
        model.train()
        T = len(obs_buf)
        obs_a  = torch.tensor(np.stack(obs_buf),  dtype=torch.float32, device=device).reshape(T * n_agents, obs_dim)
        act_a  = torch.tensor(np.stack(act_buf),  dtype=torch.long,    device=device).reshape(-1)
        logp_a = torch.tensor(np.stack(logp_buf), dtype=torch.float32, device=device).reshape(-1)
        val_a  = torch.tensor(np.stack(val_buf),  dtype=torch.float32, device=device).reshape(-1)
        rew_a  = torch.tensor(np.stack(rew_buf),  dtype=torch.float32, device=device).reshape(-1)
        done_a = torch.tensor(done_buf, dtype=torch.float32, device=device).repeat_interleave(n_agents)

        adv_a, ret_a = compute_gae(rew_a, val_a, done_a)
        adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

        n_samples = obs_a.shape[0]
        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                _, new_logp, entropy, new_val, _ = model.get_action_and_value(obs_a[mb], action=act_a[mb])
                ratio   = torch.exp(new_logp - logp_a[mb])
                pg_loss = -torch.min(ratio * adv_a[mb],
                                     torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_a[mb]).mean()
                vf_loss = F.mse_loss(new_val, ret_a[mb])
                loss    = pg_loss + VF_COEF * vf_loss - ENTROPY_COEF * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                last_loss = loss.item()

        # ── Diagnostics ────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t  = torch.tensor(obs_arr, device=device)
            _, _, Z = model(obs_t)
        last_effrank_n = effective_rank(Z) / n_agents

        logits_list = []
        with torch.no_grad():
            for i in range(n_agents):
                lgt, _, _ = model(obs_t[i:i+1])
                logits_list.append(lgt.squeeze(0))
        last_d_act = action_diversity(logits_list)

        elapsed = time.time() - train_start

        # Convergence check: EMA-smoothed loss plateau
        if ema_loss is None:
            ema_loss = last_loss
        else:
            ema_loss = CONV_EMA_ALPHA * last_loss + (1 - CONV_EMA_ALPHA) * ema_loss
        ema_history.append(ema_loss)
        if update_count >= MIN_UPDATES and len(ema_history) >= CONV_WINDOW:
            window = ema_history[-CONV_WINDOW:]
            mean_w = sum(window) / len(window)
            std_w  = (sum((v - mean_w) ** 2 for v in window) / len(window)) ** 0.5
            if std_w < CONV_STD_TOL:
                converged = True

        ema_str = f"{ema_loss:.4f}" if ema_loss is not None else "nan"
        print(f"update={update_count} steps={total_steps} elapsed={elapsed:.0f}s "
              f"effrank_n={last_effrank_n:.3f} d_act={last_d_act:.4f} "
              f"loss={last_loss:.4f} ema={ema_str}"
              + (" [converged]" if converged else ""),
              flush=True)

        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        val_buf.clear(); rew_buf.clear(); done_buf.clear()
        update_count += 1

    training_seconds = time.time() - train_start

    print(f"probe: collecting {PROBE_EVAL_EPISODES} episodes...", flush=True)
    try:
        probe_acc = compute_probe_accuracy(model, env, device)
    except Exception as e:
        print(f"probe error: {e}", flush=True)
        probe_acc = float("nan")

    env.close()

    print("---", flush=True)
    print(f"effrank_n:        {last_effrank_n:.4f}", flush=True)
    print(f"probe_acc:        {probe_acc:.4f}", flush=True)
    print(f"d_act:            {last_d_act:.4f}", flush=True)
    print(f"training_seconds: {training_seconds:.1f}", flush=True)
    print(f"num_steps:        {total_steps}", flush=True)
    print(f"num_agents:       {n_agents}", flush=True)
    print(f"reward_type:      {REWARD_TYPE}", flush=True)
    print(f"env:              smacv2_{MAP_NAME}", flush=True)
    print(f"probe_chance:     {1.0/NUM_ROLES:.3f}", flush=True)
    print(f"seed:             {SEED}", flush=True)
    print(f"converged:        {converged}", flush=True)

    # Append result to TSV
    import csv, pathlib
    result_path = pathlib.Path(RESULTS_FILE)
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["env", "reward_type", "seed", "effrank_n", "probe_acc",
                             "d_act", "num_steps", "training_seconds", "converged"])
        writer.writerow([f"smacv2_{MAP_NAME}", REWARD_TYPE, SEED,
                         f"{last_effrank_n:.4f}", f"{probe_acc:.4f}",
                         f"{last_d_act:.4f}", total_steps,
                         f"{training_seconds:.1f}", converged])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--reward-type", choices=["individual", "shared"])
    parser.add_argument("--budget",      type=int)
    parser.add_argument("--seed",        type=int)
    parser.add_argument("--n-units",     type=int)
    args = parser.parse_args()

    if args.reward_type: REWARD_TYPE          = args.reward_type
    if args.budget:      TRAIN_BUDGET_SECONDS = args.budget
    if args.seed is not None: SEED            = args.seed
    if args.n_units:     NUM_UNITS            = args.n_units

    main(dry_run=args.dry_run)
