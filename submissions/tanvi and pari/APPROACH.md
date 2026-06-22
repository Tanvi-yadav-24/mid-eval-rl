# Approach — Team "YourName and Partner"

> Replace the team name above (and the folder name) with your actual first names before submitting.

## 1. Understanding the problem

10 ads, 5,000 impressions, hidden fixed click-rates, Bernoulli rewards. The
score is total clicks averaged over 10 seeds, so the policy needs to (a)
figure out which ad is best reasonably quickly and (b) stop spending
impressions on exploration once it's confident. The top ads are close
together, so the hard part is separating the best ad from the runner-up
without overspending on ads that are clearly bad.

## 2. What we chose and why

We used **UCB1 (Upper Confidence Bound)** with a tuned exploration constant.

Each round, instead of picking the ad with the best average reward so far
(pure greedy — risks locking onto the wrong ad too early) or exploring
uniformly at random (epsilon-greedy — wastes pulls on ads we're already sure
are bad), we pick the ad with the highest "optimistic" value:

```
value[a] + c * sqrt(ln(t) / count[a])
```

The bonus term is large for ads we've tried only a few times and shrinks as
`count[a]` grows. This automatically targets exploration at the ads we're
*least* certain about, rather than spreading it uniformly — which matters
here specifically because the top few ads have nearly identical click-rates
and need more careful comparison than the obviously-bad ads.

We also force one pull of every ad at the very start so no ad's estimate
starts from zero data.

**Tuning `c`:** The textbook UCB1 bound uses `c = sqrt(2) ≈ 1.41`, which is
provably safe for *any* horizon, but that makes it more conservative than
necessary for a short, known horizon (5,000) with only 10 arms — it keeps
re-checking already-bad ads far longer than needed. We swept `c` from 1.5
down to 0.05 on the practice CTRs in `local_test.py` and found performance
peaked around `c = 0.15`, well below the textbook value, because the smaller
bonus commits to the best-looking ad faster while still being large enough
early on to tell the top few ads apart.

We also tried adding a "pure exploitation" cutoff for the last 5–15% of
impressions (always pick the current best mean, no bonus). This barely
moved the score (≤0.2 clicks difference), so we left it out to keep the
policy simple.

## 3. What else we tried

| Variant | Local avg clicks | Notes |
|---|---|---|
| Decaying epsilon-greedy (`n_arms/t`, from the example) | 496.5 | Explores uniformly at random when it does explore; can't tell the top two ads apart efficiently. |
| Thompson Sampling (`Beta(1,1)` prior, baseline) | 477.3 | Strong, but pure Beta sampling occasionally re-explores already-confident ads more than needed on this horizon. |
| UCB1, `c = sqrt(2) ≈ 1.41` (textbook) | ~383 | Way over-explores for a 5,000-step, 10-arm horizon — never commits hard enough. |
| UCB1, `c = 0.5` | 451.2 | Better, still over-exploring. |
| **UCB1, `c = 0.15` (submitted)** | **523.4** | Best balance: explores enough early to separate close ads, commits hard afterward. |

(Numbers from `local_test.py` on the practice CTRs — rerun this on your own
machine and update the numbers, since the real hidden CTRs differ.)

## 4. Results

- Local practice score: **523.4 clicks**, regret **31.0** (optimal is 550,
  random is 336.3 → closes about **87%** of the random→optimal gap).
- Best website leaderboard score: **401.3 clicks**, regret **27.6**,
  CTR **8.03%**, **rank #1 of 2**. (The real hidden best-ad CTR is lower
  than the practice set's, so the absolute click count isn't directly
  comparable to the local test — the regret and CTR numbers are.)

The site's "where your impressions went" breakdown confirms the problem
statement's warning about close CTRs: the truly-best ad (ad 3) got the most
pulls (~2,330 of 5,000), but a strong runner-up (ad 7) still pulled ~1,660 —
about a third of the budget. The cumulative-clicks-vs-oracle chart shows no
single bad early commitment; instead the gap from the oracle line grows
slowly and steadily, which is what you'd expect from UCB1 spending a
meaningful chunk of impressions distinguishing two close-performing ads
rather than from one wrong decision early on.

## 5. What we'd do next

- The impression-allocation chart shows our main inefficiency: ad 3 (truly
  best) and ad 7 (close runner-up) together absorbed most of the budget.
  A bigger or differently-shaped confidence bonus specifically in the
  regime where two ads' intervals overlap (rather than a single global `c`)
  could shave off some of that ~1,660-pull spend on ad 7 once ad 3 is
  clearly ahead.
- Try a **non-uniform Beta prior** (e.g. `Beta(1, 4)`, encoding "most ads are
  probably mediocre") combined with the UCB bonus, since a pure frequentist
  bonus and a Bayesian posterior capture slightly different information
  about uncertainty.
- Make `c` **adaptive** rather than a fixed constant — e.g. start larger and
  decay it as `t` approaches the horizon, similar in spirit to the decaying
  epsilon idea but applied to the confidence bonus instead of a random
  exploration probability.
- Run a wider seed sweep locally (more than the 10 fixed seeds) when tuning
  `c`, to make sure 0.15 isn't slightly overfit to this particular practice
  CTR set before relying on it for the real hidden CTRs.

## How to reproduce

```
cd starter
python local_test.py "../submissions/YourName and Partner/policy.py"
```
