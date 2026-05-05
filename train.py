"""
Tribal Village autoresearch training script.
Implements MAPPO with representation geometry diagnostics (EffRank/n, D_act, probe accuracy).
Fixed 10-minute wall-clock training budget for reproducible M5 Mac experiments.

The agent modifies this file. prepare_env.py is fixed and not modified.
"""

from __future__ import annotations

import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

warnings.filterwarnings("ignore")

# =============================================================================
# EXPERIMENT CONFIG — agent modifies these
# =============================================================================

REWARD_TYPE = "individual"       # "individual", "shared", or "mixed"
MIXED_INDIVIDUAL_WEIGHT = 0.2    # weight of individual reward in mixed mode (rest is shared)
USE_CONTRASTIVE = False          # add inter-agent InfoNCE loss
CONTRASTIVE_ALPHA = 6.8e-4       # contrastive loss weight
CONTRASTIVE_TEMP = 0.19          # InfoNCE temperature
NUM_AGENTS = 12                  # 6, 12, or 24
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
PPO_CLIP = 0.2
VF_COEF = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
ROLLOUT_STEPS = 512             # steps per rollout before update
ENCODER_HIDDEN = 256            # encoder hidden size
ENCODER_LAYERS = 2              # number of encoder layers

# Fixed constants (do not modify)
TRAIN_BUDGET_SECONDS = 600      # 10-minute wall-clock budget
PROBE_EVAL_STEPS = 2000         # steps to collect for probe evaluation
EFFRANK_FREQ = 20               # compute EffRank every N updates
SEED = 0

# =============================================================================
# Environment setup — from prepare_env.py logic
# =============================================================================

def make_env(num_agents: int):
    from tribal_village_env import TribalVillageEnv
    env = TribalVillageEnv(config={
        "max_steps": 1024,
        "render_mode": "ansi",
    })
    # The env always returns all agent slots; we use the first num_agents
    return env


def extract_agents(obs_dict: dict, agent_keys: list[str]) -> np.ndarray:
    """Stack per-agent observations for the active subset."""
    return np.stack([obs_dict[k] for k in agent_keys])


# =============================================================================
# Model
# =============================================================================

class SharedEncoder(nn.Module):
    def __init__(self, obs_channels: int, hidden: int, n_layers: int):
        super().__init__()
        # Flatten spatial observation into MLP input
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            *[layer for _ in range(n_layers - 1)
              for layer in [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]],
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.net(x)


class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = ENCODER_HIDDEN, n_layers: int = ENCODER_LAYERS):
        super().__init__()
        self.encoder = SharedEncoder(obs_dim, hidden, n_layers)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        z = self.encoder(obs)
        return self.actor(z), self.critic(z).squeeze(-1), z

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        logits, value, z = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, z


# =============================================================================
# Diagnostics
# =============================================================================

def effective_rank(Z: torch.Tensor) -> float:
    """SVD-based effective rank of embedding matrix Z (n_agents x d)."""
    if Z.shape[0] < 2:
        return 1.0
    Z_centered = Z - Z.mean(0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(Z_centered, full_matrices=False)
        S = S[S > 1e-8]
        if S.numel() == 0:
            return 1.0
        p = S / S.sum()
        return float(torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item())
    except Exception:
        return float("nan")


def action_diversity(logits_list: list[torch.Tensor]) -> float:
    """Mean pairwise KL divergence between agents' action distributions."""
    if len(logits_list) < 2:
        return 0.0
    probs = [F.softmax(l, dim=-1) for l in logits_list]
    kl_sum = 0.0
    count = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            kl = F.kl_div(probs[i].log(), probs[j], reduction="batchmean").item()
            kl_sum += kl
            count += 1
    return kl_sum / max(count, 1)


def inter_agent_infonce(Z: torch.Tensor, Z_future: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Inter-agent InfoNCE loss.
    Positive: same agent at a future timestep.
    Negatives: all other agents at the current timestep.
    Z: (n_agents, d) current embeddings
    Z_future: (n_agents, d) future embeddings (same agent, future step)
    """
    n = Z.shape[0]
    Z = F.normalize(Z, dim=-1)
    Z_future = F.normalize(Z_future, dim=-1)

    loss = 0.0
    for i in range(n):
        pos_sim = (Z[i] * Z_future[i]).sum() / temp
        neg_sims = torch.stack([(Z[i] * Z[j]).sum() / temp for j in range(n) if j != i])
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        label = torch.zeros(1, dtype=torch.long, device=Z.device)
        loss = loss + F.cross_entropy(logits.unsqueeze(0), label)
    return loss / n


def compute_probe_accuracy(
    model: MAPPOAgent,
    env,
    n_agents: int,
    device: torch.device,
    n_steps: int = PROBE_EVAL_STEPS,
) -> float:
    """
    Collect embeddings with individual rewards. Train a logistic regression
    to predict top-half vs. bottom-half performance rank from frozen embeddings.
    """
    from sklearn.linear_model import LogisticRegression

    model.eval()
    all_z, all_returns = [], []

    try:
        obs_raw, _ = env.reset(seed=42)
    except Exception:
        return float("nan")

    all_keys = sorted(obs_raw.keys(), key=lambda k: int(k.split("_")[1]))
    probe_keys = all_keys[:n_agents]
    obs_flat_probe = int(np.prod(obs_raw[probe_keys[0]].shape))

    agent_returns = np.zeros(n_agents)
    step_embeddings = [[] for _ in range(n_agents)]

    for _ in range(n_steps):
        obs_arr = np.stack([obs_raw[k] for k in probe_keys])
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).reshape(n_agents, -1) / 255.0
        with torch.no_grad():
            z = model.encoder(obs_t)
        for i in range(n_agents):
            step_embeddings[i].append(z[i].cpu().numpy())

        full_actions = np.zeros(env.action_space.nvec.shape[0], dtype=np.int64)
        full_actions[:n_agents] = np.random.randint(0, env.action_space.nvec[0], size=n_agents)
        obs_raw, rewards_dict, terminated, truncated, _ = env.step(full_actions)
        agent_returns += np.array([rewards_dict.get(k, 0.0) for k in probe_keys])
        if terminated or truncated:
            try:
                obs_raw, _ = env.reset()
            except Exception:
                break

    # Build labels: top-half vs bottom-half by total return
    threshold = np.median(agent_returns)
    labels = (agent_returns > threshold).astype(int)

    # Build feature matrix: mean embedding per agent
    X = np.array([np.mean(step_embeddings[i], axis=0) for i in range(n_agents)])
    if np.std(labels) == 0:
        return 0.5

    try:
        clf = LogisticRegression(max_iter=1000)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X, labels, cv=min(3, n_agents // 2))
        return float(scores.mean())
    except Exception:
        return float("nan")


# =============================================================================
# GAE computation
# =============================================================================

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:len(rewards)]
    return advantages, returns


# =============================================================================
# Main training loop
# =============================================================================

def main(dry_run: bool = False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    env = make_env(NUM_AGENTS)
    obs_raw, _ = env.reset(seed=SEED)

    # Determine active agents and observation size
    all_agent_keys = sorted(obs_raw.keys(), key=lambda k: int(k.split("_")[1]))
    agent_keys = all_agent_keys[:NUM_AGENTS]
    single_obs = obs_raw[agent_keys[0]]
    obs_flat = int(np.prod(single_obs.shape))
    n_actions = int(env.action_space.nvec[0])

    model = MAPPOAgent(obs_flat, n_actions, ENCODER_HIDDEN, ENCODER_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    if dry_run:
        print("Dry run OK. obs_flat=%d n_actions=%d device=%s n_agents=%d" % (obs_flat, n_actions, device, NUM_AGENTS))
        env.close()
        return

    print("Starting training: agents=%d reward=%s contrastive=%s device=%s" % (
        NUM_AGENTS, REWARD_TYPE, USE_CONTRASTIVE, device))

    train_start = time.time()
    total_steps = 0
    update_count = 0
    last_effrank_n = float("nan")
    last_d_act = float("nan")
    effrank_history = []

    # Rollout buffers
    obs_buf = []
    act_buf = []
    logp_buf = []
    val_buf = []
    rew_buf = []
    done_buf = []
    emb_buf = []         # for contrastive: store embeddings

    obs_cur = extract_agents(obs_raw, agent_keys)

    while True:
        elapsed = time.time() - train_start
        if elapsed >= TRAIN_BUDGET_SECONDS:
            break

        # --- Rollout ---
        model.eval()
        for _ in range(ROLLOUT_STEPS):
            obs_t = torch.tensor(obs_cur, dtype=torch.float32, device=device).reshape(NUM_AGENTS, -1) / 255.0
            with torch.no_grad():
                acts, logps, _, vals, z = model.get_action_and_value(obs_t)

            full_actions = np.zeros(env.action_space.nvec.shape[0], dtype=np.int64)
            full_actions[:NUM_AGENTS] = acts.cpu().numpy()

            obs_next_raw, rewards_dict, terminated, truncated, _ = env.step(full_actions)
            agent_rewards = np.array([rewards_dict.get(k, 0.0) for k in agent_keys])

            # Reward shaping by condition
            if REWARD_TYPE == "shared":
                r = np.full(NUM_AGENTS, agent_rewards.mean())
            elif REWARD_TYPE == "mixed":
                r = MIXED_INDIVIDUAL_WEIGHT * agent_rewards + (1 - MIXED_INDIVIDUAL_WEIGHT) * agent_rewards.mean()
            else:  # individual
                r = agent_rewards

            obs_buf.append(obs_cur.copy())
            act_buf.append(acts.cpu().numpy())
            logp_buf.append(logps.cpu().numpy())
            val_buf.append(vals.cpu().numpy())
            rew_buf.append(r)
            done_buf.append(float(terminated or truncated))
            emb_buf.append(z.detach().cpu())

            obs_cur = extract_agents(obs_next_raw, agent_keys)
            total_steps += NUM_AGENTS

            if terminated or truncated:
                obs_raw, _ = env.reset()
                obs_cur = extract_agents(obs_raw, agent_keys)

        # --- Update ---
        model.train()
        obs_arr = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device).reshape(-1, obs_flat) / 255.0
        act_arr = torch.tensor(np.array(act_buf), dtype=torch.long, device=device).reshape(-1)
        logp_arr = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device).reshape(-1)
        val_arr = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device).reshape(-1)
        rew_arr = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device).reshape(-1)
        done_arr = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device).reshape(ROLLOUT_STEPS)

        # Expand done and val per agent
        done_expanded = done_arr.repeat_interleave(NUM_AGENTS)
        adv_arr, ret_arr = compute_gae(rew_arr, val_arr, done_expanded)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Contrastive: pair embeddings[t] with embeddings[t+1]
        emb_stack = torch.stack(emb_buf, dim=0)  # (T, n_agents, d)

        n_samples = obs_arr.shape[0]
        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                _, new_logp, entropy, new_val, new_z = model.get_action_and_value(obs_arr[mb], act_arr[mb])
                ratio = torch.exp(new_logp - logp_arr[mb])
                pg_loss = -torch.min(
                    ratio * adv_arr[mb],
                    torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_arr[mb]
                ).mean()
                vf_loss = F.mse_loss(new_val, ret_arr[mb])
                ent_loss = -entropy.mean()

                loss = pg_loss + VF_COEF * vf_loss + ENTROPY_COEF * ent_loss

                # Contrastive loss (inter-agent InfoNCE)
                if USE_CONTRASTIVE and emb_stack.shape[0] > 1:
                    t_idx = torch.randint(0, emb_stack.shape[0] - 1, (1,)).item()
                    Z_cur = emb_stack[t_idx].to(device)
                    Z_fut = emb_stack[t_idx + 1].to(device)
                    loss = loss + CONTRASTIVE_ALPHA * inter_agent_infonce(Z_cur, Z_fut, CONTRASTIVE_TEMP)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # EffRank diagnostic
        if update_count % EFFRANK_FREQ == 0:
            with torch.no_grad():
                sample_obs = torch.tensor(obs_cur, dtype=torch.float32, device=device).reshape(NUM_AGENTS, -1) / 255.0
                _, _, _, _, Z = model.get_action_and_value(sample_obs)
            er = effective_rank(Z)
            last_effrank_n = er / NUM_AGENTS
            effrank_history.append(last_effrank_n)

            # Action diversity
            logits_list = []
            with torch.no_grad():
                for i in range(NUM_AGENTS):
                    lgt, _, _, _, _ = model.get_action_and_value(sample_obs[i:i+1])
                    logits_list.append(lgt.squeeze(0))
            last_d_act = action_diversity(logits_list)

        # Clear buffers
        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        val_buf.clear(); rew_buf.clear(); done_buf.clear(); emb_buf.clear()
        update_count += 1

    training_seconds = time.time() - train_start

    # Probe accuracy
    try:
        probe_acc = compute_probe_accuracy(model, env, NUM_AGENTS, device)
    except Exception:
        probe_acc = float("nan")

    env.close()

    # Print summary in autoresearch format
    print("---")
    print(f"effrank_n:        {last_effrank_n:.4f}")
    print(f"probe_acc:        {probe_acc:.4f}")
    print(f"d_act:            {last_d_act:.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {time.time() - train_start + training_seconds:.1f}")
    print(f"num_steps:        {total_steps}")
    print(f"num_agents:       {NUM_AGENTS}")
    print(f"reward_type:      {REWARD_TYPE}")
    print(f"use_contrastive:  {USE_CONTRASTIVE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Check env loads, then exit")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
