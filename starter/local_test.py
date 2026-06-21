"""
local_test.py  --  Test your policy.py OFFLINE before you upload it.

    python local_test.py                # tests ./policy.py
    python local_test.py mypolicy.py    # tests a specific file

This mirrors how the real leaderboard scores you (same loop, same averaging over
many fixed seeds), so the numbers here are directly comparable in *spirit*.

IMPORTANT: the click-rates below are PRACTICE values we made up for you. The real
contest uses different, hidden click-rates — so don't hard-code anything! If your
policy does well here for the right reasons (it actually learns), it'll do well
there too.
"""

import importlib.util
import random
import sys

import numpy as np

# --- practice problem (NOT the real hidden one) ------------------------------
PRACTICE_CTRS = [0.05, 0.09, 0.04, 0.11, 0.107, 0.02, 0.07, 0.10, 0.03, 0.06]
HORIZON = 10_000
SEEDS = list(range(30))


def load_policy(path):
    spec = importlib.util.spec_from_file_location("student_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Policy"):
        raise SystemExit("ERROR: your file must define a class named `Policy`.")
    return mod.Policy


def run_episode(PolicyClass, ctrs, horizon, seed):
    n_arms = len(ctrs)
    best = max(ctrs)
    env_rng = np.random.default_rng(seed)
    np.random.seed(seed)          # so randomised policies are reproducible
    random.seed(seed)

    policy = PolicyClass(n_arms, horizon)
    clicks, regret = 0, 0.0
    for _ in range(horizon):
        arm = int(policy.select_arm())
        assert 0 <= arm < n_arms, f"select_arm() returned out-of-range arm {arm}"
        reward = 1 if env_rng.random() < ctrs[arm] else 0
        policy.update(arm, reward)
        clicks += reward
        regret += best - ctrs[arm]
    return clicks, regret


def score(PolicyClass, ctrs):
    cs, rs = [], []
    for s in SEEDS:
        c, r = run_episode(PolicyClass, ctrs, HORIZON, s)
        cs.append(c)
        rs.append(r)
    return np.mean(cs), np.mean(rs)


class _Random:
    def __init__(self, n_arms, horizon):
        self.n = n_arms
    def select_arm(self):
        return random.randrange(self.n)
    def update(self, arm, reward):
        pass


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "policy.py"
    PolicyClass = load_policy(path)

    optimal = max(PRACTICE_CTRS) * HORIZON
    rnd_clicks, _ = score(_Random, PRACTICE_CTRS)
    your_clicks, your_regret = score(PolicyClass, PRACTICE_CTRS)

    print("=" * 48)
    print(f"  Testing: {path}")
    print(f"  Practice problem: {len(PRACTICE_CTRS)} ads, {HORIZON} impressions,"
          f" {len(SEEDS)} seeds")
    print("=" * 48)
    print(f"  Optimal (always best ad) : {optimal:8.1f} clicks")
    print(f"  Random baseline          : {rnd_clicks:8.1f} clicks")
    print(f"  >> YOUR POLICY <<        : {your_clicks:8.1f} clicks")
    print(f"     regret (lower better) : {your_regret:8.1f}")
    print("-" * 48)
    if your_clicks <= rnd_clicks + 1:
        print("  Hmm, you're not beating random. Is your policy learning?")
    elif your_clicks >= optimal - 0.05 * optimal:
        print("  Excellent — you're hugging the optimal line!")
    else:
        frac = (your_clicks - rnd_clicks) / (optimal - rnd_clicks)
        print(f"  You closed {100*frac:.0f}% of the gap from random to optimal.")
    print("=" * 48)
