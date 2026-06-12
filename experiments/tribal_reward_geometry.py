"""Tribal Village reward-geometry experiment (paper Table 1).

Studies how feedback attribution (individual / mixed / shared rewards) shapes the
geometry of a *shared* MAPPO encoder, with a division-of-labor task design so that
the three fixed roles (agent_id % 3) correspond to distinct chain stages
(gather / craft / deposit). The natural environment reward keeps behavior
meaningful; role-aligned potential-based shaping gives the roles their meaning.

Two safeguards (requested):
  FM#1 (meaningful behavior): every trained policy is scored against no-op and
        random baselines on the *natural* environment return and on real chain
        events (gather/craft/deposit). Representation metrics for a run are only
        trustworthy if it clears the behavior gate; the verdict is logged, never
        hidden.
  FM#2 (consistent eval/classifier): all conditions are scored by the *same*
        condition-agnostic code in canonical_geometry.py (fixed i%3 role probe,
        4-fold agent-generalization CV, chance = 1/3; ordered-KL D_act; EffRank/n).
        The only thing that differs across conditions is the training shared_frac.

Run one (condition, seed) per process (the Nim FFI uses a single global env):

    python experiments/tribal_reward_geometry.py --shared_frac 0.0 --seed 0 \
        --total_steps 4000000 --wandb_project rl_workshop_2026
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from canonical_geometry import (  # noqa: E402
    DEFAULT_ALTAR_LAYER,
    DEFAULT_GOLD_LAYER,
    effective_rank,
    fixed_role_probe_accuracy,
    mix_rewards,
    ordered_kl_action_diversity,
    role_labels,
    role_probe_chance,
)

LIB_PATH = REPO / "tribal_village_env" / "libtribal_village.so"
ACTION_SPACE_SIZE = 308          # 11 verbs x 28 args
GOLD_LAYER = DEFAULT_GOLD_LAYER   # 26
MAGMA_LAYER = 30                  # ThingMagmaLayer (smelting station for craft)
ALTAR_LAYER = DEFAULT_ALTAR_LAYER  # 31
ROLE_TARGET_LAYER = {0: GOLD_LAYER, 1: MAGMA_LAYER, 2: ALTAR_LAYER}


# ───────────────────────── Nim config struct ──────────────────────────────────

class NimConfig(ctypes.Structure):
    _fields_ = [
        ("max_steps", ctypes.c_int32),
        ("victory_condition", ctypes.c_int32),
        ("tumor_spawn_rate", ctypes.c_float),
        ("heart_reward", ctypes.c_float),
        ("ore_reward", ctypes.c_float),
        ("bar_reward", ctypes.c_float),
        ("wood_reward", ctypes.c_float),
        ("water_reward", ctypes.c_float),
        ("wheat_reward", ctypes.c_float),
        ("spear_reward", ctypes.c_float),
        ("armor_reward", ctypes.c_float),
        ("food_reward", ctypes.c_float),
        ("cloth_reward", ctypes.c_float),
        ("tumor_kill_reward", ctypes.c_float),
        ("survival_penalty", ctypes.c_float),
        ("death_penalty", ctypes.c_float),
    ]


# Explicit, dense, learnable reward coefficients (the env's nan defaults must be
# replaced). The gold->bar->heart chain is the highest-value path; secondary
# resource collection keeps the signal dense enough to bootstrap from scratch.
REWARD_CFG = dict(
    max_steps=1000,
    victory_condition=0,
    tumor_spawn_rate=0.0,
    heart_reward=3.0,
    ore_reward=0.05,
    bar_reward=0.5,
    wood_reward=0.05,
    water_reward=0.05,
    wheat_reward=0.05,
    spear_reward=0.2,
    armor_reward=0.2,
    food_reward=0.2,
    cloth_reward=0.2,
    tumor_kill_reward=0.5,
    survival_penalty=-0.001,
    death_penalty=-1.0,
)


# ───────────────────────── ctypes env wrapper ─────────────────────────────────

class TribalEnv:
    """Single Tribal Village env (the Nim FFI keeps one global env)."""

    def __init__(self, max_steps: int = 1000):
        self.lib = ctypes.CDLL(str(LIB_PATH))
        self._setup()
        self.num_agents = self.lib.tribal_village_get_num_agents()
        self.obs_layers = self.lib.tribal_village_get_obs_layers()
        self.obs_w = self.lib.tribal_village_get_obs_width()
        self.obs_h = self.lib.tribal_village_get_obs_height()
        self.obs_dim = self.obs_layers * self.obs_w * self.obs_h
        self.n_actions = ACTION_SPACE_SIZE
        self.max_steps = max_steps

        self.env = self.lib.tribal_village_create()
        if not self.env:
            raise RuntimeError("tribal_village_create failed")
        self._apply_config()

        n = self.num_agents
        self.obs = np.zeros((n, self.obs_layers, self.obs_w, self.obs_h), np.uint8)
        self.rew = np.zeros(n, np.float32)
        self.term = np.zeros(n, np.float32)
        self.trunc = np.zeros(n, np.float32)
        self.stage = np.zeros((n, 3), np.float32)
        self.step_count = 0

    def _setup(self):
        L = self.lib
        L.tribal_village_create.restype = ctypes.c_void_p
        L.tribal_village_set_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(NimConfig)]
        L.tribal_village_set_config.restype = ctypes.c_int32
        for fn in ("get_num_agents", "get_obs_layers", "get_obs_width", "get_obs_height"):
            getattr(L, f"tribal_village_{fn}").restype = ctypes.c_int32
        L.tribal_village_reset_and_get_obs.argtypes = [ctypes.c_void_p] * 5
        L.tribal_village_reset_and_get_obs.restype = ctypes.c_int32
        L.tribal_village_step_with_pointers.argtypes = [ctypes.c_void_p] * 6
        L.tribal_village_step_with_pointers.restype = ctypes.c_int32
        L.tribal_village_get_stage_events.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.tribal_village_get_stage_events.restype = ctypes.c_int32

    def _apply_config(self):
        cfg = NimConfig(**REWARD_CFG)
        self.lib.tribal_village_set_config(self.env, ctypes.byref(cfg))

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.c_void_p)

    def reset(self):
        self.lib.tribal_village_reset_and_get_obs(
            self.env, self._p(self.obs), self._p(self.rew), self._p(self.term), self._p(self.trunc)
        )
        self.step_count = 0
        return self.obs.copy()

    def step(self, actions: np.ndarray):
        a = actions.astype(np.uint16)
        self.lib.tribal_village_step_with_pointers(
            self.env, self._p(a), self._p(self.obs), self._p(self.rew), self._p(self.term), self._p(self.trunc)
        )
        self.lib.tribal_village_get_stage_events(self.env, self._p(self.stage))
        self.step_count += 1
        done = bool(np.all(self.term > 0.5)) or self.step_count >= self.max_steps
        out = (self.obs.copy(), self.rew.copy(), self.stage.copy(), done)
        if done:
            self.reset()
        return out


# Direction arg (0-7) -> (dx, dy) in egocentric grid (y increases downward).
_DIR_OFFSETS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0),
                4: (-1, -1), 5: (1, -1), 6: (-1, 1), 7: (1, 1)}
USE_VERB = 3  # action = USE_VERB*28 + direction interacts with the adjacent tile


def _role_target_layers(n: int) -> np.ndarray:
    labels = role_labels(n)
    return np.array([ROLE_TARGET_LAYER[int(r)] for r in labels])


def role_target_adjacent_dir(obs: np.ndarray) -> np.ndarray:
    """For each agent, the direction (0-7) to an adjacent (dist-1) tile of its
    role's target layer, or -1 if none adjacent (vectorized over agents)."""
    n = obs.shape[0]
    cy, cx = obs.shape[2] // 2, obs.shape[3] // 2
    tl = _role_target_layers(n)
    role_plane = obs[np.arange(n), tl]                     # (n, H, W) each agent's target layer
    out = np.full(n, -1, np.int64)
    for d in range(7, -1, -1):                              # 8 fixed iterations (not over agents)
        dx, dy = _DIR_OFFSETS[d]
        present = role_plane[:, cy + dy, cx + dx] > 0
        out[present] = d
    return out


def role_potential_vec(obs: np.ndarray, dist_map: np.ndarray, far: float) -> np.ndarray:
    """Per-agent negative distance to nearest role-target tile (vectorized)."""
    n = obs.shape[0]
    tl = _role_target_layers(n)
    role_plane = obs[np.arange(n), tl] > 0                  # (n, H, W)
    big = dist_map[None] + (~role_plane) * 1e6
    mind = big.reshape(n, -1).min(axis=1)
    return -np.where(mind > 1e5, far, mind)


def role_potential(obs: np.ndarray) -> np.ndarray:
    """Per-agent negative distance to its role's target tile within the egocentric
    11x11 view. Agent sits at the view center. Returns shape (num_agents,)."""
    n = obs.shape[0]
    cy, cx = obs.shape[2] // 2, obs.shape[3] // 2
    yy, xx = np.mgrid[0:obs.shape[2], 0:obs.shape[3]]
    dist = np.abs(yy - cy) + np.abs(xx - cx)          # manhattan distance map
    far = float(dist.max() + 1)
    labels = role_labels(n)
    phi = np.empty(n, np.float64)
    for i in range(n):
        layer = ROLE_TARGET_LAYER[int(labels[i])]
        mask = obs[i, layer] > 0
        phi[i] = -float(dist[mask].min()) if mask.any() else -far
    return phi


# ───────────────────────────── policy ─────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared encoder for all agents: 2-layer MLP (256, LayerNorm, ReLU) -> 256-d
    embedding, then linear actor/critic heads (matches the paper architecture)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, emb: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, emb), nn.LayerNorm(emb), nn.ReLU(),
        )
        self.actor = nn.Linear(emb, n_actions)
        self.critic = nn.Linear(emb, 1)

    def forward(self, obs):
        z = self.encoder(obs)
        return self.actor(z), self.critic(z).squeeze(-1), z

    def act(self, obs, action=None):
        logits, value, z = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, z, logits


def compute_gae(rew, val, done, next_val, gamma=0.99, lam=0.95):
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = 0.0
    for t in reversed(range(T)):
        nt = 1.0 - done[t]
        nv = next_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * nt - val[t]
        adv[t] = last = delta + gamma * lam * nt * last
    return adv, adv + val


# ───────────────────────────── evaluation ─────────────────────────────────────

@torch.no_grad()
def rollout_collect(env, policy, device, n_steps, deterministic):
    """Run a policy for n_steps; return per-step embeddings, logits, natural reward
    sum, and chain-stage event totals. Used identically for eval and for the
    no-op/random baselines (FM#2: one code path for every condition)."""
    obs = env.reset()
    n = env.num_agents
    embs, logits_all = [], []
    nat_return = 0.0
    stage_tot = np.zeros(3)
    for _ in range(n_steps):
        if policy is None:
            acts = np.zeros(n, np.int64)            # no-op baseline
        elif policy == "random":
            acts = np.random.randint(0, ACTION_SPACE_SIZE, size=n)
        else:
            ot = torch.tensor(obs.reshape(n, -1), dtype=torch.float32, device=device) / 255.0
            logits, _, z = policy.forward(ot)
            acts = (logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()).cpu().numpy()
            embs.append(z.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
        obs, rew, stage, _ = env.step(acts)
        nat_return += float(rew.sum())
        stage_tot += stage.sum(0)
    # task_score reflects real task progress: env reward + weighted chain work.
    task_score = nat_return + 0.1 * stage_tot[0] + 0.5 * stage_tot[1] + 2.0 * stage_tot[2]
    result = {"nat_return": nat_return, "stage": stage_tot, "task_score": task_score}
    if embs:
        result["embeddings"] = np.stack(embs)       # [steps, agents, emb]
        result["logits"] = np.stack(logits_all)     # [steps, agents, actions]
    return result


def evaluate(env, policy, device, eval_steps, n_agents):
    """Condition-agnostic metric computation (FM#2)."""
    coll = rollout_collect(env, policy, device, eval_steps, deterministic=True)
    embs = coll["embeddings"]                        # [steps, agents, emb]
    logits = coll["logits"]                          # [steps, agents, actions]
    flat_emb = embs.reshape(-1, embs.shape[-1])
    eff_rank = effective_rank(flat_emb)
    d_act = ordered_kl_action_diversity(logits)
    probe_acc, probe_meta = fixed_role_probe_accuracy(embs)
    return {
        "effrank": eff_rank,
        "effrank_per_agent": eff_rank / n_agents,
        "d_act": d_act,
        "probe_accuracy": probe_acc,
        "probe_chance": probe_meta["chance"],
        "probe_lift": probe_acc - probe_meta["chance"],
        "probe_fold_scores": probe_meta["fold_scores"],
        "eval_nat_return": coll["nat_return"],
        "eval_task_score": coll["task_score"],
        "eval_stage_gather": float(coll["stage"][0]),
        "eval_stage_craft": float(coll["stage"][1]),
        "eval_stage_deposit": float(coll["stage"][2]),
    }


def behavior_gate(env, policy, device, eval_steps, trained_score, trained_stage):
    """FM#1: trained policy must beat no-op and random baselines on the task score
    (env reward + real chain work) by a clear margin. Never hides a failure."""
    noop = rollout_collect(env, None, device, eval_steps, deterministic=True)
    rand = rollout_collect(env, "random", device, eval_steps, deterministic=True)
    baseline = max(noop["task_score"], rand["task_score"])
    margin = abs(baseline) * 0.10 + 1.0
    passed = bool(trained_score > baseline + margin)
    return {
        "gate_passed": passed,
        "trained_task_score": trained_score,
        "noop_task_score": noop["task_score"],
        "random_task_score": rand["task_score"],
        "gate_margin": margin,
        "trained_chain_events": float(np.sum(trained_stage)),
        "random_chain_events": float(np.sum(rand["stage"])),
    }


# ───────────────────────────── training ───────────────────────────────────────

def train(args):
    import wandb

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)   # avoid 256-core thread thrash under heavy concurrency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_frac = args.shared_frac
    cond = {0.0: "individual", 0.8: "mixed", 1.0: "shared"}.get(shared_frac, f"frac{shared_frac}")
    run_name = args.run_name or f"tv_{cond}_seed{args.seed}"

    env = TribalEnv(max_steps=args.max_steps)
    n = env.num_agents
    obs_dim = env.obs_dim
    policy = ActorCritic(obs_dim, env.n_actions, args.hidden, args.emb).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name,
               config={**vars(args), "condition": cond, "num_agents": n,
                       "reward_cfg": REWARD_CFG}, mode=args.wandb_mode, reinit=True)

    T = args.num_steps
    # Precompute the egocentric manhattan distance map once (vectorized shaping).
    cy, cx = env.obs_h // 2, env.obs_w // 2
    yy, xx = np.mgrid[0:env.obs_h, 0:env.obs_w]
    DIST_MAP = (np.abs(yy - cy) + np.abs(xx - cx)).astype(np.float64)
    FAR = float(DIST_MAP.max() + 1)
    obs = env.reset()
    prev_phi = role_potential_vec(obs, DIST_MAP, FAR)
    gamma, lam, beta = args.gamma, args.gae_lambda, args.shaping_coef

    obs_buf = np.zeros((T, n, obs_dim), np.float32)
    act_buf = np.zeros((T, n), np.int64)
    logp_buf = np.zeros((T, n), np.float32)
    rew_buf = np.zeros((T, n), np.float32)
    val_buf = np.zeros((T, n), np.float32)
    done_buf = np.zeros(T, np.float32)

    global_step = 0
    recent_nat, recent_stage = [], np.zeros(3)
    t0 = time.time()
    update = 0

    while global_step < args.total_steps:
        for t in range(T):
            adj_dir = role_target_adjacent_dir(obs)   # role target adjacent in the acted-on state
            ot = torch.tensor(obs.reshape(n, -1), dtype=torch.float32, device=device) / 255.0
            with torch.no_grad():
                act, logp, _, val, _, _ = policy.act(ot)
            acts = act.cpu().numpy()
            nobs, rew, stage, done = env.step(acts)

            # Division-of-labor reward. The env's ore_reward is dead code, so the
            # gather->craft->deposit chain is rewarded directly from the stage-event
            # counters (dense + meaningful), plus a role-specific bonus on each
            # agent's own stage (specialization), plus role-aligned potential
            # shaping (guidance), then mixed by attribution granularity.
            phi = role_potential_vec(nobs, DIST_MAP, FAR)
            shaping = beta * (gamma * phi - prev_phi)
            prev_phi = phi
            role_idx = role_labels(n)
            own_stage = stage[np.arange(n), role_idx]                 # this agent's role-stage events
            chain = 0.1 * stage[:, 0] + 0.5 * stage[:, 1] + 2.0 * stage[:, 2]  # shared chain value
            # Use-affordance bonus: reward emitting `use` toward an adjacent role
            # target (teaches the gather/craft/deposit action so the chain fires).
            used_correct = (adj_dir >= 0) & (acts == USE_VERB * 28 + np.maximum(adj_dir, 0))
            affordance = args.affordance_coef * used_correct.astype(np.float64)
            individual = (rew.astype(np.float64) + chain
                          + args.role_bonus * own_stage + affordance + shaping)
            mixed = mix_rewards(individual, shared_frac)

            obs_buf[t] = obs.reshape(n, -1)
            act_buf[t] = acts
            logp_buf[t] = logp.cpu().numpy()
            rew_buf[t] = mixed
            val_buf[t] = val.cpu().numpy()
            done_buf[t] = float(done)

            recent_nat.append(float(rew.sum()))
            recent_stage += stage.sum(0)
            obs = nobs
            global_step += n

        # bootstrap + per-agent GAE
        ot = torch.tensor(obs.reshape(n, -1), dtype=torch.float32, device=device) / 255.0
        with torch.no_grad():
            _, _, _, next_val, _, _ = policy.act(ot)
        next_val = next_val.cpu().numpy()

        adv = np.zeros_like(rew_buf)
        ret = np.zeros_like(rew_buf)
        done_t = torch.tensor(done_buf, dtype=torch.float32)
        for a in range(n):
            ad, rt = compute_gae(
                torch.tensor(rew_buf[:, a]), torch.tensor(val_buf[:, a]), done_t,
                torch.tensor(float(next_val[a])), gamma, lam)
            adv[:, a] = ad.numpy()
            ret[:, a] = rt.numpy()

        b_obs = torch.tensor(obs_buf.reshape(T * n, obs_dim), device=device) / 255.0
        b_act = torch.tensor(act_buf.reshape(T * n), device=device)
        b_logp = torch.tensor(logp_buf.reshape(T * n), device=device)
        b_adv = torch.tensor(adv.reshape(T * n), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret.reshape(T * n), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idx = np.arange(T * n)
        pg = vl = ent_l = 0.0
        nb = 0
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, T * n, args.minibatch_size):
                mb = idx[s:s + args.minibatch_size]
                _, nlogp, ent, nval, _, _ = policy.act(b_obs[mb], b_act[mb])
                ratio = torch.exp(nlogp - b_logp[mb])
                p1 = -b_adv[mb] * ratio
                p2 = -b_adv[mb] * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                pgl = torch.max(p1, p2).mean()
                vll = F.mse_loss(nval, b_ret[mb])
                loss = pgl + 0.5 * vll - args.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                pg += pgl.item(); vl += vll.item(); ent_l += ent.mean().item(); nb += 1

        update += 1
        if update % args.log_interval == 0:
            sps = global_step / max(1e-6, time.time() - t0)
            mean_nat = float(np.mean(recent_nat[-2000:])) if recent_nat else 0.0
            wandb.log({
                "global_step": global_step,
                "charts/natural_reward_per_step": mean_nat,
                "charts/chain_gather": float(recent_stage[0]),
                "charts/chain_craft": float(recent_stage[1]),
                "charts/chain_deposit": float(recent_stage[2]),
                "charts/sps": sps,
                "losses/policy": pg / max(1, nb),
                "losses/value": vl / max(1, nb),
                "losses/entropy": ent_l / max(1, nb),
            }, step=global_step)
            print(f"[{run_name}] step={global_step:,} nat/step={mean_nat:.3f} "
                  f"chain(g/c/d)={recent_stage.astype(int)} sps={sps:.0f}", flush=True)
            recent_stage = np.zeros(3)

        if args.metric_interval and update % args.metric_interval == 0:
            # Periodic geometry metrics -> learning curve of the representation.
            m = evaluate(env, policy, device, max(200, args.eval_steps // 2), n)
            wandb.log({f"curve/{k}": v for k, v in m.items()
                       if isinstance(v, (int, float))}, step=global_step)
            print(f"[{run_name}] curve step={global_step:,} probe={m['probe_accuracy']:.3f} "
                  f"effrank/n={m['effrank_per_agent']:.3f} d_act={m['d_act']:.4f}", flush=True)

        if args.ckpt_interval and update % args.ckpt_interval == 0:
            _save_ckpt(policy, run_name, global_step)

    # ---- final eval + behavior gate (identical code for every condition) ----
    print(f"[{run_name}] training done; evaluating...", flush=True)
    metrics = evaluate(env, policy, device, args.eval_steps, n)
    gate = behavior_gate(env, policy, device, args.eval_steps,
                         metrics["eval_task_score"],
                         np.array([metrics["eval_stage_gather"], metrics["eval_stage_craft"],
                                   metrics["eval_stage_deposit"]]))
    ckpt = _save_ckpt(policy, run_name, global_step)

    result = {"run_name": run_name, "condition": cond, "shared_frac": shared_frac,
              "seed": args.seed, "total_steps": global_step, "checkpoint": ckpt,
              **metrics, **gate}
    wandb.log({f"final/{k}": v for k, v in result.items() if isinstance(v, (int, float, bool))})
    wandb.summary.update({k: v for k, v in result.items() if isinstance(v, (int, float, bool))})

    out = REPO / "results" / f"{run_name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[{run_name}] RESULT {json.dumps({k: v for k, v in result.items() if isinstance(v,(int,float,bool))}, indent=2)}", flush=True)
    print(f"[{run_name}] gate_passed={gate['gate_passed']} probe={metrics['probe_accuracy']:.3f} "
          f"effrank/n={metrics['effrank_per_agent']:.3f} d_act={metrics['d_act']:.4f}", flush=True)
    wandb.finish()
    return result


def _save_ckpt(policy, run_name, step):
    d = REPO / "checkpoints" / run_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step}.pt"
    torch.save({"model": policy.state_dict(), "step": step}, path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_frac", type=float, required=True, help="0.0 individual, 0.8 mixed, 1.0 shared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_steps", type=int, default=4_000_000)
    ap.add_argument("--num_steps", type=int, default=256)     # rollout: 256 x 12 = 3072 agent-steps
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--minibatch_size", type=int, default=1024)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--shaping_coef", type=float, default=0.1)
    ap.add_argument("--role_bonus", type=float, default=1.0, help="extra reward per own-stage event (division of labor)")
    ap.add_argument("--affordance_coef", type=float, default=0.2, help="bonus for `use` toward an adjacent role target")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--eval_steps", type=int, default=800)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--ckpt_interval", type=int, default=200)
    ap.add_argument("--metric_interval", type=int, default=40, help="updates between periodic geometry-metric logging")
    ap.add_argument("--torch_threads", type=int, default=2, help="torch intra-op threads (low avoids thrash under concurrency)")
    ap.add_argument("--wandb_project", type=str, default="rl_workshop_2026")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online")
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
