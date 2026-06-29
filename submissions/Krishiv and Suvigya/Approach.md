Here's a more natural, human-written version of the report:

---

# Our Approach

## The Problem

We were given a 10-armed Bernoulli bandit with 5000 impressions to work with. Each arm is an ad with an unknown click-through rate, and the goal is simple: get as many clicks as possible. The catch, of course, is the classic explore-exploit tradeoff — you need to learn which ad is best, but every impression spent learning is one not spent earning clicks.

---

# What We Built

Our final solution is an adaptive ε-greedy algorithm that treats the 5000-impression horizon in three distinct phases rather than using a fixed exploration rate throughout.

The core idea is straightforward: exploration should be heavy early on, taper off in the middle, and stop entirely once you're confident enough in your estimates.

### Phase 1 — Explore Hard (first 20%)

Early on, we set ε = 1/t^0.2. At the very start this is close to 1, so almost every pick is exploratory. As impressions accumulate, it decays naturally, letting us start leaning on whichever arm looks best without fully committing yet.

### Phase 2 — Pull Back (20–40%)

Once we've seen enough, we switch to ε = 0.5/t^0.3 — a noticeably lower exploration rate. We're exploiting most of the time now, but leaving the door open to catch any early estimation mistakes.

### Phase 3 — Go All In (after 40%)

Past the 40% mark, exploration stops completely. We just keep picking the arm with the highest estimated CTR for the rest of the horizon. In a stationary environment, there's no reason to keep exploring once you've had enough data to form a reliable picture.

---

# Keeping Track of CTRs

We use an online mean update to track each arm's estimated click-through rate:

Q_new = Q_old + (R − Q_old) / N

It's memory-efficient and gives you the exact sample mean without storing every observation.

---

# What Else We Tried

Before landing on this, we worked through the usual suspects:

**ε-Greedy (baseline)** — Simple and transparent, but a fixed ε is wasteful; you're still randomly exploring even when you've basically figured out which arm is best.

**UCB1** — Solid theoretical guarantees, but in practice it was a bit conservative on our specific environment.

**Thompson Sampling** — Generally outperformed both ε-greedy and UCB1, which wasn't surprising. We spent a fair amount of time here.

**Thompson Sampling variants** — We tried Jeffreys priors (Beta(0.5, 0.5)), Beta(2, 2), optimistic initialization, Bayesian confidence bonuses, forced exploration, and hybrid UCB–Thompson hybrids. Some were competitive, but none beat the adaptive ε-greedy consistently.

---

# Results

Our adaptive ε-greedy policy outperformed everything else we tested and landed on the practice leaderboard with a score of around **388.8**.

---

# Why This Works

The insight is simple but easy to overlook: not all impressions are equal. Early ones are for learning. Middle ones are for refining. Late ones should almost entirely be for cashing in. A fixed exploration rate treats all 5000 impressions the same, which is a poor fit for a finite horizon.

By explicitly splitting the horizon into three phases, we get fast learning up front and maximum exploitation when it matters most — in the back half of the run where most of the clicks are actually earned.