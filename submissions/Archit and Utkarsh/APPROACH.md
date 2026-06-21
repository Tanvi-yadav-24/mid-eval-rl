# Approach — Team "Archit and Utkarsh" (EXAMPLE write-up)

> This is a sample to show the depth we expect. Your real `APPROACH.md` should
> describe what *you* actually did. Replace everything below.

## 1. Understanding the problem
10 ads, 10,000 impressions, hidden fixed click-rates, Bernoulli rewards. The
score is total clicks averaged over 30 seeds, so we want a policy that (a) finds
the best ad quickly and (b) doesn't keep wasting impressions once it's confident.
Because the top ads are close together, the hard part is telling the **best** ad
apart from the **second-best** without overspending.

## 2. What we chose and why
We used **epsilon-greedy with a decaying epsilon** (`epsilon = n_arms / t`):
- Early on, `t` is small, so epsilon is high → we explore aggressively and get a
  rough estimate of every ad.
- As `t` grows, epsilon shrinks → we increasingly exploit the best-looking ad.

We also force one pull of every ad up front so no ad starts with a blind estimate.
We chose a decaying schedule over a fixed epsilon because a fixed epsilon keeps
paying an exploration tax forever — with 10,000 rounds that tax adds up.

## 3. What else we tried
| Variant | Local avg clicks | Notes |
|---|---|---|
| Fixed epsilon = 0.1 | ~9XX | Bleeds clicks late; never stops exploring. |
| Explore-then-commit (explore 50/arm) | ~9XX | Sensitive to the wrong commit; unlucky seeds hurt. |
| **Decaying epsilon (submitted)** | **~10XX** | Best balance for us. |
| Thompson Sampling (baseline) | ~10XX | Strong; we got close but want to revisit priors. |

(Numbers above are from `local_test.py` on the practice CTRs — replace with yours.)

## 4. Results
- Local practice score: **~10XX clicks** (closed ~78% of the random→optimal gap).
- Best website leaderboard score: **<fill in>**, rank **<fill in>**.

## 5. What we'd do next
Add a **UCB1**-style confidence bonus so exploration targets the *uncertain* ads
specifically, instead of exploring uniformly at random. We think that would help
most exactly where this problem is hard: separating the top two or three ads.

## How to reproduce
```bash
cd starter
python local_test.py "../submissions/Archit and Utkarsh/policy.py"
```
