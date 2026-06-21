# 📜 Problem Statement — The Ad Heist

## The setup
You are the brain behind a single ad slot. Every time a user arrives, you must
pick **one of 10 ads** to show. Each ad has a fixed but **hidden**
click-through-rate (CTR) — the probability a user clicks it. You show an ad and
the user either clicks (**reward = 1**) or doesn't (**reward = 0**).

You get **10,000 impressions** in total. The catch: the only way to learn an ad's
CTR is to *spend impressions* on it. Explore too much on weak ads and you waste
clicks; commit too early and you may crown the wrong ad for the rest of the run.
This is the classic **explore-vs-exploit** trade-off — a **stationary Bernoulli
multi-armed bandit**.

| Property | Value |
|---|---|
| Number of ads (arms) | **10** |
| Impressions (horizon) | **10,000** |
| Reward | **Bernoulli** — click (1) or no-click (0) |
| Click-rates | **fixed** for the whole run, but **hidden** from you |
| Scoring | total clicks, **averaged over 30 fixed seeds** |

> The top few ads have **very close** click-rates on purpose — telling the best
> ad apart from the runner-up takes genuine, careful exploration.

## What you build
A single file `policy.py` with a class named exactly `Policy`:

```python
class Policy:
    def __init__(self, n_arms, horizon):
        # n_arms  = number of ads (= 10)
        # horizon = total impressions you'll get (= 10000)
        ...

    def select_arm(self):
        # return the index of the ad to show now: an int in [0, n_arms)
        ...

    def update(self, arm, reward):
        # arm    = the ad you just showed
        # reward = 1 if clicked, else 0
        ...
```

The evaluator runs exactly this loop against the hidden CTRs:

```python
policy = Policy(n_arms, horizon)
total_clicks = 0
for t in range(horizon):
    arm = policy.select_arm()
    reward = show_ad(arm)        # 1 with probability CTR[arm], else 0
    policy.update(arm, reward)
    total_clicks += reward       # <- this is your score
```

This whole run is repeated on **30 fixed random seeds** (identical for every team)
and your score is the **average total clicks**. We also report:
- **Regret** — clicks you missed vs. always showing the single best ad (lower is better).
- **CTR%** — your clicks ÷ impressions.

## Rules
- Class must be named exactly `Policy` with the three methods above.
- `select_arm()` must return an integer in `[0, 10)`.
- Only `numpy` + the Python standard library. No internet, file I/O, or extra packages.
- No hard-coding ad indices or CTRs — they're hidden and differ per seed. **Learn** them.
- Keep per-round work light (you have a few-second budget for the whole run).
- **Any** algorithm is allowed — you are not limited to what was taught in class.

## How to test & submit
- **Offline:** `python starter/local_test.py path/to/policy.py` (uses practice CTRs).
- **Live:** upload `policy.py` on the website to get your real score + rank.
- **Final/graded:** open a Pull Request — see [README.md](README.md) for the exact steps.
