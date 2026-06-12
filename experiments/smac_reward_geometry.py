"""SMACv2 boundary-condition experiment (paper Table 2).

A solved benchmark with published win-rate baselines, used to show the proposed
metrics (EffRank/n, D_act) behave sensibly on competent agents. SMACv2 encodes
unit type directly in the observation, so EffRank/n should saturate regardless of
reward attribution while D_act still responds — the paper's complementary-coverage
claim.

Reward attribution (same axis as Tribal Village):
  individual : each agent's reward = team damage share + its OWN ally-health delta
  shared     : every agent gets the team-averaged of that quantity

Metrics use the SAME consistent core as Tribal Village
(experiments/canonical_geometry.py): EffRank/n, ordered-KL D_act, and a unit-type
probe (marine/marauder/medivac) with held-out CV.

    python experiments/smac_reward_geometry.py --reward_type individual --seed 0 \
        --total_steps 2000000 --wandb_project rl_workshop_2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from canonical_geometry import effective_rank, ordered_kl_action_diversity  # noqa: E402

if "SC2PATH" not in os.environ:
    for cand in (Path.home() / "StarCraftII", REPO / "StarCraftII"):
        if cand.is_dir():
            os.environ["SC2PATH"] = str(cand)
            break

from smacv2.env import StarCraftCapabilityEnvWrapper  # noqa: E402


TERRAN_UNITS = ["marine", "marauder", "medivac"]
UNIT_TO_LABEL = {u: i for i, u in enumerate(TERRAN_UNITS)}


def make_env(seed: int, n_units: int = 6):
    """10gen_terran capability env: random terran team compositions of n_units."""
    distribution_config = {
        "n_units": n_units,
        "n_enemies": n_units,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": TERRAN_UNITS,
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": n_units,
            "map_x": 32,
            "map_y": 32,
        },
    }
    return StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=False,
        conic_fov=False,
        use_unit_ranges=True,
        min_attack_range=2,
        obs_own_pos=True,
        seed=seed,
    )


class SMACEnv:
    """Single SMACv2 env with per-agent health tracking for individual rewards."""

    def __init__(self, seed: int, n_units: int = 6):
        self.env = make_env(seed, n_units)
        info = self.env.get_env_info()
        self.n_agents = info["n_agents"]
        self.n_actions = info["n_actions"]
        self.obs_dim = info["obs_shape"]
        self.episode_limit = info["episode_limit"]
        self.steps = 0
        self._prev_ally_health = None

    def _ally_health(self) -> np.ndarray:
        h = np.zeros(self.n_agents, np.float64)
        for i in range(self.n_agents):
            u = self.env.env.agents.get(i)
            if u is not None and u.health_max > 0:
                h[i] = u.health / u.health_max
        return h

    def unit_labels(self) -> np.ndarray:
        e = self.env.env
        type_to_label = {e.marine_id: 0, e.marauder_id: 1, e.medivac_id: 2}
        labels = np.zeros(self.n_agents, np.int64)
        for i in range(self.n_agents):
            u = e.agents.get(i)
            labels[i] = type_to_label.get(u.unit_type, 0) if u is not None else 0
        return labels

    def reset(self):
        self.env.reset()
        self.steps = 0
        self._prev_ally_health = self._ally_health()
        return self._obs(), self._avail()

    def _obs(self):
        return np.array([self.env.get_obs_agent(a) for a in range(self.n_agents)], np.float32)

    def _avail(self):
        inner = self.env.env
        return np.array([inner.get_avail_agent_actions(a) for a in range(self.n_agents)], np.float32)

    def step(self, actions, reward_type):
        # Safety clamp against the INNER env's avail (the same source the SMAC
        # step assertion uses), replacing any invalid action with the first
        # available one (no-op for dead agents).
        inner = self.env.env
        actions = np.asarray(actions).copy()
        for i in range(self.n_agents):
            avail = inner.get_avail_agent_actions(i)
            if actions[i] >= len(avail) or avail[int(actions[i])] == 0:
                actions[i] = int(np.argmax(avail))
        try:
            team_reward, terminated, info = self.env.step(actions.tolist())
        except AssertionError:
            # Extremely rare SMAC avail/step mismatch: end the episode cleanly.
            team_reward, terminated, info = 0.0, True, {"battle_won": False}
        self.steps += 1
        health = self._ally_health()
        delta = health - self._prev_ally_health      # per-agent ally-health delta
        self._prev_ally_health = health
        share = team_reward / self.n_agents
        individual = share + delta                    # team damage share + own survival
        if reward_type == "shared":
            rew = np.full(self.n_agents, float(individual.mean()), np.float64)
        else:
            rew = individual
        done = bool(terminated) or self.steps >= self.episode_limit
        return self._obs(), self._avail(), rew, done, float(team_reward), bool(info.get("battle_won", False))

    def close(self):
        self.env.close()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256, emb=256):
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

    def act(self, obs, avail, action=None):
        logits, value, z = self.forward(obs)
        logits = logits.masked_fill(avail == 0, -1e10)
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


def unit_type_probe(embeddings, labels):
    """Held-out CV logistic regression: unit type from frozen embeddings.
    embeddings [steps, agents, emb], labels [steps, agents]."""
    X = embeddings.reshape(-1, embeddings.shape[-1])
    y = labels.reshape(-1)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float(max(counts) / counts.sum()), 1.0 / max(1, len(classes))
    cv = int(min(4, counts.min()))
    if cv < 2:
        return float("nan"), 1.0 / len(classes)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, random_state=0))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    chance = float(counts.max() / counts.sum())
    return float(scores.mean()), chance


@torch.no_grad()
def evaluate(env, policy, device, n_episodes):
    embs, logits_all, labels_all = [], [], []
    wins, rets = [], []
    for _ in range(n_episodes):
        obs, avail = env.reset()
        labels = env.unit_labels()
        done = False
        ep_team = 0.0
        while not done:
            ot = torch.tensor(obs, dtype=torch.float32, device=device)
            at = torch.tensor(avail, dtype=torch.float32, device=device)
            logits, _, z = policy.forward(ot)
            logits = logits.masked_fill(at == 0, -1e10)
            acts = logits.argmax(-1).cpu().numpy()
            embs.append(z.cpu().numpy()); logits_all.append(logits.cpu().numpy()); labels_all.append(labels)
            obs, avail, rew, done, team_r, won = env.step(acts, "individual")
            ep_team += team_r
        wins.append(float(won)); rets.append(ep_team)
    embs = np.stack(embs); logits = np.stack(logits_all); labels = np.stack(labels_all)
    eff = effective_rank(embs.reshape(-1, embs.shape[-1]))
    d_act = ordered_kl_action_diversity(logits)
    probe, chance = unit_type_probe(embs, labels)
    return {
        "effrank": eff, "effrank_per_agent": eff / env.n_agents,
        "d_act": d_act, "probe_accuracy": probe, "probe_chance": chance,
        "win_rate": float(np.mean(wins)), "eval_return": float(np.mean(rets)),
    }


def train(args):
    import wandb
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"smac_{args.reward_type}_seed{args.seed}"
    env = SMACEnv(args.seed, args.n_units)
    n, obs_dim, n_act = env.n_agents, env.obs_dim, env.n_actions
    policy = ActorCritic(obs_dim, n_act, args.hidden, args.emb).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name,
               config={**vars(args), "n_agents": n}, mode=args.wandb_mode, reinit=True)

    T = args.num_steps
    obs, avail = env.reset()
    gstep = 0; upd = 0; t0 = time.time()
    recent_win, recent_ret = [], []
    while gstep < args.total_steps:
        ob = np.zeros((T, n, obs_dim), np.float32); av = np.zeros((T, n, n_act), np.float32)
        ac = np.zeros((T, n), np.int64); lp = np.zeros((T, n), np.float32)
        rw = np.zeros((T, n), np.float32); vl = np.zeros((T, n), np.float32); dn = np.zeros(T, np.float32)
        for t in range(T):
            ot = torch.tensor(obs, dtype=torch.float32, device=device)
            at = torch.tensor(avail, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, l, _, v, _, _ = policy.act(ot, at)
            acts = a.cpu().numpy()
            ob[t] = obs; av[t] = avail; ac[t] = acts
            lp[t] = l.cpu().numpy(); vl[t] = v.cpu().numpy()
            nobs, navail, rew, done, team_r, won = env.step(acts, args.reward_type)
            rw[t] = rew; dn[t] = float(done)
            obs, avail = nobs, navail
            if done:
                recent_win.append(float(won)); recent_ret.append(team_r)
                obs, avail = env.reset()
            gstep += n
        ot = torch.tensor(obs, dtype=torch.float32, device=device)
        at = torch.tensor(avail, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, _, nv, _, _ = policy.act(ot, at)
        nv = nv.cpu().numpy()
        adv = np.zeros_like(rw); ret = np.zeros_like(rw); dt = torch.tensor(dn)
        for ag in range(n):
            ad, rt = compute_gae(torch.tensor(rw[:, ag]), torch.tensor(vl[:, ag]), dt,
                                 torch.tensor(float(nv[ag])), args.gamma, args.gae_lambda)
            adv[:, ag] = ad.numpy(); ret[:, ag] = rt.numpy()
        bo = torch.tensor(ob.reshape(T * n, obs_dim), device=device)
        ba = torch.tensor(av.reshape(T * n, n_act), device=device)
        bac = torch.tensor(ac.reshape(T * n), device=device)
        blp = torch.tensor(lp.reshape(T * n), device=device)
        bad = torch.tensor(adv.reshape(T * n), dtype=torch.float32, device=device)
        bret = torch.tensor(ret.reshape(T * n), dtype=torch.float32, device=device)
        bad = (bad - bad.mean()) / (bad.std() + 1e-8)
        idx = np.arange(T * n)
        pg = vlo = ent = 0.0; nb = 0
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, T * n, args.minibatch_size):
                mb = idx[s:s + args.minibatch_size]
                _, nlp, en, nval, _, _ = policy.act(bo[mb], ba[mb], bac[mb])
                ratio = torch.exp(nlp - blp[mb])
                pgl = torch.max(-bad[mb] * ratio, -bad[mb] * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                vll = F.mse_loss(nval, bret[mb])
                loss = pgl + 0.5 * vll - args.ent_coef * en.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
                pg += pgl.item(); vlo += vll.item(); ent += en.mean().item(); nb += 1
        upd += 1
        if upd % args.log_interval == 0:
            wr = float(np.mean(recent_win[-50:])) if recent_win else 0.0
            sps = gstep / max(1e-6, time.time() - t0)
            wandb.log({"global_step": gstep, "charts/win_rate": wr,
                       "charts/team_return": float(np.mean(recent_ret[-50:])) if recent_ret else 0.0,
                       "charts/sps": sps, "losses/policy": pg / max(1, nb),
                       "losses/value": vlo / max(1, nb), "losses/entropy": ent / max(1, nb)}, step=gstep)
            print(f"[{run_name}] step={gstep:,} win_rate={wr:.3f} sps={sps:.0f}", flush=True)
        if args.metric_interval and upd % args.metric_interval == 0:
            m = evaluate(env, policy, device, args.eval_episodes)
            wandb.log({f"curve/{k}": v for k, v in m.items()}, step=gstep)
            print(f"[{run_name}] curve step={gstep:,} win={m['win_rate']:.2f} "
                  f"effrank/n={m['effrank_per_agent']:.3f} d_act={m['d_act']:.4f} probe={m['probe_accuracy']:.3f}", flush=True)

    metrics = evaluate(env, policy, device, max(args.eval_episodes, 20))
    d = REPO / "checkpoints" / run_name; d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "step": gstep}, d / f"step_{gstep}.pt")
    result = {"run_name": run_name, "reward_type": args.reward_type, "seed": args.seed,
              "total_steps": gstep, **metrics}
    wandb.log({f"final/{k}": v for k, v in metrics.items()})
    wandb.summary.update(result)
    (REPO / "results" / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_name}] RESULT {json.dumps(result, indent=2)}", flush=True)
    wandb.finish(); env.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_type", choices=["individual", "shared"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_units", type=int, default=6)
    ap.add_argument("--total_steps", type=int, default=2_000_000)
    ap.add_argument("--num_steps", type=int, default=256)
    ap.add_argument("--minibatch_size", type=int, default=1024)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--metric_interval", type=int, default=40)
    ap.add_argument("--torch_threads", type=int, default=2)
    ap.add_argument("--wandb_project", type=str, default="rl_workshop_2026")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online")
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
