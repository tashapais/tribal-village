"""Recompute EffRank/n the paper's way: over per-agent time-averaged centroids.

The paper's teaser defines EffRank/n on per-agent centroids (one vector per
agent), so the effective rank is <= n_agents and EffRank/n <= 1. The training
harnesses logged effrank over ALL timestep x agent embeddings (<= embedding_dim),
which is a different scale. This script reloads each checkpoint, runs a short
deterministic eval, and reports BOTH definitions so the paper table can use the
comparable one.

    python experiments/recompute_centroid.py --mode tribal
    python experiments/recompute_centroid.py --mode smac
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from canonical_geometry import effective_rank  # noqa: E402


def centroid_effrank_per_agent(embs: np.ndarray, n_agents: int) -> float:
    """embs: [steps, n_agents, d] -> effrank of per-agent centroids / n_agents."""
    centroids = embs.mean(axis=0)               # [n_agents, d]
    return effective_rank(centroids) / n_agents


def latest_ckpt(run_name: str) -> str | None:
    cks = sorted(glob.glob(str(REPO / "checkpoints" / run_name / "step_*.pt")),
                 key=lambda p: int(p.split("step_")[-1].split(".pt")[0]))
    return cks[-1] if cks else None


def recompute_tribal():
    from tribal_reward_geometry import TribalEnv, ActorCritic, rollout_collect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TribalEnv(max_steps=1000)
    out = {}
    for cond in ("individual", "mixed", "shared"):
        for seed in range(3):
            run = f"tv_{cond}_seed{seed}"
            ck = latest_ckpt(run)
            if not ck:
                continue
            policy = ActorCritic(env.obs_dim, env.n_actions, 256, 256).to(device)
            policy.load_state_dict(torch.load(ck, map_location=device)["model"])
            policy.eval()
            coll = rollout_collect(env, policy, device, 800, deterministic=True)
            embs = coll["embeddings"]            # [steps, agents, d]
            out[run] = centroid_effrank_per_agent(embs, env.num_agents)
            print(f"{run}: EffRank/n(centroid)={out[run]:.4f}", flush=True)
    return out


def recompute_smac():
    from smac_reward_geometry import SMACEnv, ActorCritic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for rt in ("individual", "shared"):
        for seed in range(3):
            run = f"smac_{rt}_seed{seed}"
            ck = latest_ckpt(run)
            if not ck:
                continue
            env = SMACEnv(seed, 6)
            policy = ActorCritic(env.obs_dim, env.n_actions, 256, 256).to(device)
            policy.load_state_dict(torch.load(ck, map_location=device)["model"])
            policy.eval()
            embs = []
            for _ in range(16):
                obs, avail = env.reset()
                done = False
                while not done:
                    with torch.no_grad():
                        logits, _, z = policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                    at = torch.tensor(avail, dtype=torch.float32, device=device)
                    acts = logits.masked_fill(at == 0, -1e10).argmax(-1).cpu().numpy()
                    embs.append(z.cpu().numpy())
                    obs, avail, _, done, _, _ = env.step(acts, "individual")
            env.close()
            arr = np.stack(embs)                 # [steps, agents, d]
            out[run] = centroid_effrank_per_agent(arr, env.n_agents)
            print(f"{run}: EffRank/n(centroid)={out[run]:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tribal", "smac"], required=True)
    args = ap.parse_args()
    out = recompute_tribal() if args.mode == "tribal" else recompute_smac()
    path = REPO / "results" / f"centroid_effrank_{args.mode}.json"
    path.write_text(json.dumps(out, indent=2))
    # aggregate
    import re
    groups: dict[str, list] = {}
    for run, v in out.items():
        cond = re.sub(r"_seed\d+", "", run)
        groups.setdefault(cond, []).append(v)
    print("\n=== EffRank/n (per-agent centroid) ===")
    for cond, vals in groups.items():
        a = np.array(vals)
        print(f"{cond}: {a.mean():.4f} ± {a.std():.4f}")


if __name__ == "__main__":
    main()
