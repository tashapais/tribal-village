"""
Generate SMACv2 PCA embedding figure matching the teaser style.
Trains individual and shared conditions (seed 0) to convergence,
collects embeddings during eval, then plots.

Output: samples/figures/pca_smacv2_individual_vs_shared.png
"""

from __future__ import annotations
import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

warnings.filterwarnings("ignore")
os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── Config ────────────────────────────────────────────────────────────────────
NUM_UNITS      = 6
NUM_ROLES      = 3
SEED           = 0
LR             = 3e-4
ENTROPY_COEF   = 0.01
PPO_CLIP       = 0.2
VF_COEF        = 0.5
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
UPDATE_EPOCHS  = 4
MINIBATCH_SIZE = 256
ROLLOUT_STEPS  = 128
HIDDEN         = 256
N_LAYERS       = 2
MAP_NAME       = "10gen_terran"
BUDGET_SECONDS = 900   # 15-min hard cap per condition
CONV_EMA_ALPHA = 0.1
CONV_WINDOW    = 30
CONV_STD_TOL   = 0.03
MIN_UPDATES    = 80
N_EVAL_STEPS   = 600   # steps to collect embeddings for PCA

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../698bd9a65fba5c04a962e794/samples/figures/pca_smacv2_individual_vs_shared.png",
)

# Role colors and labels matching teaser style
ROLE_COLORS = ["#e8706a", "#6ab187", "#6a9fe8"]   # red, green, blue
ROLE_LABELS = ["Marine (Attacker)", "Marauder (Heavy)", "Medivac (Healer)"]

# ── Environment helpers ───────────────────────────────────────────────────────
def make_env(seed=0):
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    cfg = {
        "n_units": NUM_UNITS, "n_enemies": NUM_UNITS,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": [],
            "weights": [1/3, 1/3, 1/3],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5, "n_enemies": NUM_UNITS, "map_x": 32, "map_y": 32,
        },
    }
    return StarCraftCapabilityEnvWrapper(
        capability_config=cfg, map_name=MAP_NAME, debug=False,
        conic_fov=False, obs_own_pos=True, use_unit_ranges=True,
        min_attack_range=2, step_mul=8, seed=seed,
    )

def get_role_labels(env):
    inner = env.env
    role_map = {inner.marine_id: 0, inner.marauder_id: 1, inner.medivac_id: 2}
    return np.array([role_map.get(inner.agents[i].unit_type, 0) for i in range(env.n_agents)], dtype=np.int64)

def get_obs(env):
    return np.stack(env.get_obs()).astype(np.float32)

def get_avail(env):
    return np.stack([env.get_avail_agent_actions(i) for i in range(env.n_agents)]).astype(np.float32)

def get_individual_rewards(env, actions):
    inner = env.env
    n = env.n_agents
    rewards = np.zeros(n, dtype=np.float32)
    neg_scale = inner.reward_negative_scale
    n_no_attack = inner.n_actions_no_attack
    is_terran = inner.map_type in ["MMM", "terran_gen"]
    enemy_attackers, ally_healers = {}, {}
    for a_id, action in enumerate(actions):
        unit = inner.agents[a_id]
        if unit.health == 0 or action < n_no_attack:
            continue
        target_id = action - n_no_attack
        if is_terran and unit.unit_type == inner.medivac_id:
            ally_healers.setdefault(target_id, []).append(a_id)
        else:
            enemy_attackers.setdefault(target_id, []).append(a_id)
    for e_id, ids in enemy_attackers.items():
        prev = inner.previous_enemy_units.get(e_id)
        curr = inner.enemies.get(e_id)
        if prev is None or curr is None:
            continue
        dmg = max(0.0, (prev.health + prev.shield) - ((curr.health + curr.shield) if curr.health > 0 else 0.0))
        kill = inner.reward_death_value if (curr.health == 0 and prev.health > 0) else 0.0
        credit = (dmg + kill) / len(ids)
        for a in ids:
            rewards[a] += credit
    for al_id, ids in ally_healers.items():
        prev = inner.previous_ally_units.get(al_id)
        curr = inner.agents.get(al_id)
        if prev is None or curr is None:
            continue
        heal = max(0.0, curr.health - prev.health) / len(ids)
        for a in ids:
            rewards[a] += heal
    for a_id in range(n):
        prev = inner.previous_ally_units.get(a_id)
        curr = inner.agents.get(a_id)
        if prev and curr and prev.health > 0 and curr.health == 0:
            rewards[a_id] -= inner.reward_death_value * neg_scale
    if inner.reward_scale and inner.max_reward > 0:
        rewards /= inner.max_reward / inner.reward_scale_rate
    return rewards

# ── Model ─────────────────────────────────────────────────────────────────────
class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.actor   = nn.Linear(hidden, n_actions)
        self.critic  = nn.Linear(hidden, 1)

    def forward(self, obs):
        z = self.encoder(obs.float())
        return self.actor(z), self.critic(z).squeeze(-1), z

    def act(self, obs, avail=None, action=None):
        logits, value, z = self(obs)
        if avail is not None:
            logits = logits - 1e10 * (1 - avail)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, z

# ── GAE ───────────────────────────────────────────────────────────────────────
def gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nv = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv, adv + values[:len(rewards)]

# ── Diagnostics ───────────────────────────────────────────────────────────────
def effective_rank(Z):
    if Z.shape[0] < 2:
        return 1.0
    Zc = Z - Z.mean(0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(Zc, full_matrices=False)
        S = S[S > 1e-8]
        if S.numel() == 0:
            return 1.0
        p = S / S.sum()
        return float(torch.exp(-(p * torch.log(p + 1e-10)).sum()).item())
    except Exception:
        return float("nan")

def d_act(model, obs_t, avail_t, device):
    probs = []
    with torch.no_grad():
        for i in range(obs_t.shape[0]):
            logits, _, _ = model(obs_t[i:i+1])
            av = avail_t[i:i+1]
            logits = logits - 1e10 * (1 - av)
            p = F.softmax(logits.squeeze(0), dim=-1).clamp(min=1e-8)
            probs.append(p / p.sum())
    kl, count = 0.0, 0
    for i in range(len(probs)):
        for j in range(i+1, len(probs)):
            kl += (probs[j] * (probs[j].log() - probs[i].log())).sum().item()
            count += 1
    return kl / max(count, 1)

# ── Training + embedding collection ──────────────────────────────────────────
def train_and_collect(reward_type: str, device):
    print(f"\n{'='*60}", flush=True)
    print(f"Training: reward={reward_type}", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    env = make_env(SEED)
    env.reset()
    n_agents  = env.n_agents
    obs_dim   = env.get_obs_size()
    n_actions = env.get_total_actions()

    model     = MAPPOAgent(obs_dim, n_actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-5)

    ema_loss, ema_hist, converged = None, [], False
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    obs_arr = get_obs(env)
    t0 = time.time()
    update_count = 0

    while True:
        if time.time() - t0 >= BUDGET_SECONDS or converged:
            break

        model.eval()
        for _ in range(ROLLOUT_STEPS):
            obs_t   = torch.tensor(obs_arr, device=device)
            avail_t = torch.tensor(get_avail(env), device=device)
            with torch.no_grad():
                acts, logps, _, vals, _ = model.act(obs_t, avail=avail_t)
            actions = acts.cpu().numpy().tolist()
            team_r, terminated, _ = env.step(actions)

            if reward_type == "shared":
                r = np.full(n_agents, team_r, dtype=np.float32)
            elif reward_type == "mixed":
                ind_r = get_individual_rewards(env, actions)
                r = 0.2 * ind_r + 0.8 * team_r
            else:
                r = get_individual_rewards(env, actions)

            obs_buf.append(obs_arr.copy())
            act_buf.append(np.array(actions, dtype=np.int64))
            logp_buf.append(logps.cpu().numpy())
            val_buf.append(vals.cpu().numpy())
            rew_buf.append(r)
            done_buf.append(float(terminated))
            if terminated:
                env.reset()
            obs_arr = get_obs(env)

        model.train()
        T = len(obs_buf)
        obs_a  = torch.tensor(np.stack(obs_buf),  dtype=torch.float32, device=device).reshape(T*n_agents, obs_dim)
        act_a  = torch.tensor(np.stack(act_buf),  dtype=torch.long,    device=device).reshape(-1)
        logp_a = torch.tensor(np.stack(logp_buf), dtype=torch.float32, device=device).reshape(-1)
        val_a  = torch.tensor(np.stack(val_buf),  dtype=torch.float32, device=device).reshape(-1)
        rew_a  = torch.tensor(np.stack(rew_buf),  dtype=torch.float32, device=device).reshape(-1)
        done_a = torch.tensor(done_buf,            dtype=torch.float32, device=device).repeat_interleave(n_agents)
        adv_a, ret_a = gae(rew_a, val_a, done_a)
        adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

        last_loss = float("nan")
        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(obs_a.shape[0], device=device)
            for s in range(0, obs_a.shape[0], MINIBATCH_SIZE):
                mb = idx[s:s+MINIBATCH_SIZE]
                _, nlp, ent, nv, _ = model.act(obs_a[mb], action=act_a[mb])
                ratio   = torch.exp(nlp - logp_a[mb])
                pg_loss = -torch.min(ratio*adv_a[mb], torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP)*adv_a[mb]).mean()
                vf_loss = F.mse_loss(nv, ret_a[mb])
                loss    = pg_loss + VF_COEF*vf_loss - ENTROPY_COEF*ent.mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                last_loss = loss.item()

        ema_loss = CONV_EMA_ALPHA*last_loss + (1-CONV_EMA_ALPHA)*ema_loss if ema_loss is not None else last_loss
        ema_hist.append(ema_loss)
        if update_count >= MIN_UPDATES and len(ema_hist) >= CONV_WINDOW:
            w = ema_hist[-CONV_WINDOW:]
            mu = sum(w)/len(w)
            std = (sum((v-mu)**2 for v in w)/len(w))**0.5
            if std < CONV_STD_TOL:
                converged = True

        obs_t   = torch.tensor(obs_arr, device=device)
        avail_t = torch.tensor(get_avail(env), device=device)
        with torch.no_grad():
            _, _, Z = model(obs_t)
        er = effective_rank(Z) / n_agents
        da = d_act(model, obs_t, avail_t, device)

        elapsed = time.time() - t0
        print(f"  update={update_count} elapsed={elapsed:.0f}s effrank_n={er:.3f} d_act={da:.4f} ema={ema_loss:.4f}"
              + (" [converged]" if converged else ""), flush=True)

        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        val_buf.clear(); rew_buf.clear(); done_buf.clear()
        update_count += 1

    # ── Collect embeddings for PCA ────────────────────────────────────────
    print(f"  Collecting {N_EVAL_STEPS} eval steps...", flush=True)
    model.eval()
    env.reset()
    role_labels = get_role_labels(env)
    obs_arr = get_obs(env)

    all_Z, all_roles = [], []
    agent_Z = [[] for _ in range(n_agents)]  # per-agent for centroids
    da_vals, er_vals = [], []

    for _ in range(N_EVAL_STEPS):
        obs_t   = torch.tensor(obs_arr, device=device)
        avail_t = torch.tensor(get_avail(env), device=device)
        with torch.no_grad():
            _, _, Z = model(obs_t)
        Z_np = Z.cpu().numpy()
        for i in range(n_agents):
            all_Z.append(Z_np[i])
            all_roles.append(role_labels[i])
            agent_Z[i].append(Z_np[i])

        da_vals.append(d_act(model, obs_t, avail_t, device))
        er_vals.append(effective_rank(Z) / n_agents)

        acts_t = Categorical(logits=model(obs_t)[0] - 1e10*(1-avail_t)).sample()
        actions = acts_t.cpu().numpy().tolist()
        _, terminated, _ = env.step(actions)
        if terminated:
            env.reset()
            role_labels = get_role_labels(env)
        obs_arr = get_obs(env)

    env.close()

    mean_er = float(np.mean(er_vals))
    mean_da = float(np.mean(da_vals))
    print(f"  eval: effrank_n={mean_er:.3f}  d_act={mean_da:.4f}", flush=True)

    centroids = np.array([np.mean(agent_Z[i], axis=0) for i in range(n_agents)])
    return (np.array(all_Z), np.array(all_roles), centroids, role_labels,
            mean_er, mean_da)

# ── PCA plot ─────────────────────────────────────────────────────────────────
CONDITIONS = [
    ("individual", "Individual rewards"),
    ("mixed",      "Mixed (80% shared)"),
    ("shared",     "Shared rewards"),
]
PANEL_LABELS = "abc"

def make_pca_figure(all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    fig.patch.set_facecolor("#ffffff")

    for ax_i, ((reward_type, title), data) in enumerate(zip(CONDITIONS, all_data)):
        all_Z, all_roles, centroids, agent_roles, er, da = data
        ax = axes[ax_i]
        ax.set_facecolor("#ffffff")

        combined = np.vstack([all_Z, centroids])
        pca = PCA(n_components=2)
        pca.fit(combined)
        Z2    = pca.transform(all_Z)
        cent2 = pca.transform(centroids)
        var   = pca.explained_variance_ratio_

        for role_id in range(NUM_ROLES):
            mask = all_roles == role_id
            ax.scatter(Z2[mask, 0], Z2[mask, 1],
                       color=ROLE_COLORS[role_id], alpha=0.22, s=12,
                       linewidths=0, label=ROLE_LABELS[role_id])

        for i in range(len(centroids)):
            role_id = int(agent_roles[i])
            ax.scatter(cent2[i, 0], cent2[i, 1],
                       color=ROLE_COLORS[role_id], s=160, zorder=5,
                       edgecolors="black", linewidths=1.6)

        da_str = f"{da:.3f}" if not (da != da) else "n/a"
        metric_str = f"EffRank/n={er:.3f}  D_act={da_str}"
        ax.set_title(f"({PANEL_LABELS[ax_i]}) {title}\n{metric_str}",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(f"PC 1 ({var[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC 2 ({var[1]*100:.1f}%)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.85,
                      markerscale=1.4, handletextpad=0.4)

    fig.suptitle(r"SMACv2 (10gen\_terran): geometry saturates, $D_\mathrm{act}$ tracks attribution",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}", flush=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    all_data = []
    for reward_type, _ in CONDITIONS:
        all_data.append(train_and_collect(reward_type, device))

    print("\nGenerating PCA figure...", flush=True)
    make_pca_figure(all_data)
