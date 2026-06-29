# APPROACH.md

## 1. Problem Understanding

**Task:** Stationary Bernoulli multi-armed bandit — 10 ads, 5,000 impressions, hidden
fixed click-through rates (CTRs). Score = total clicks averaged over 10 fixed seeds.

**Key challenge:** The top ads have *very close* CTRs (e.g., 0.110 vs 0.107 in the
practice problem — a gap of only 0.003). Telling the best arm apart from the
second-best requires many observations, but spending too many impressions exploring
forfeits revenue.

**Information-theoretic lower bound on needed samples:** To distinguish Bernoulli(p)
from Bernoulli(p + Δ) with error probability δ, one needs roughly
`n ≥ log(1/δ) / (2Δ²)` samples per arm. With Δ = 0.003, this is ~55,000 per arm —
far more than the 500 our budget allows. Therefore, the strategy must accept some
residual uncertainty and focus on separating the *top arm from the rest* (where
gaps are larger), rather than perfectly ranking the top two.

---

## 2. Algorithms Evaluated

I compared six algorithms systematically, testing each on both the 10 practice seeds
(used by `local_test.py`) and 100 randomly chosen seeds (for robustness):

| Algorithm | Practice (10 seeds) | 100 seeds | Notes |
|---|---|---|---|
| Random baseline | 336.3 | ~305 | Floor |
| Thompson Sampling (Beta(1,1)) | 477.3 | ~468 | Provided baseline |
| Decaying ε-greedy (C=5) | 531.7 | 508.4 | ε = min(1, 5K/t) |
| UCB1 (α=2) | 398.9 | ~390 | Too much exploration |
| UCB (α=0.02) | 536.4 | 517.2 | Tuned coefficient |
| **MOSS (c=0.16)** | **536.6** | **520.2** | **Chosen — most robust** |
| MOSS (c=0.10) | 541.1 | 512.4 | Best on practice, high variance |
| KL-UCB | 475.3 | ~460 | Over-explores for T=5000 |
| Successive Elimination | 336.5 | ~336 | Wrong objective for cumulative reward |

---

## 3. Chosen Algorithm: MOSS

**MOSS** (Minimax Optimal Strategy in the Stochastic case) uses the index:

```
UCB(a) = mean(a) + c · sqrt( max(0, log(T / (K · n_a))) / n_a )
```

where:
- `T` = horizon (5,000)
- `K` = number of arms (10)
- `n_a` = number of times arm `a` has been pulled
- `c` = tuned coefficient (0.16)

### Why MOSS?

**1. Theoretically motivated.** MOSS achieves minimax-optimal regret `O(sqrt(KT))`
for the stochastic bandit problem, as proven by Audibert & Bubeck (2009). Standard
UCB1 achieves `O(sqrt(KT log T))` — the extra `log T` factor matters over 5,000
rounds.

**2. Natural exploration-to-exploitation transition.** The log term
`log(T / (K · n_a))` equals zero when `n_a = T/K`. For T=5000 and K=10, this
threshold is 500 pulls per arm. Once any arm accumulates enough evidence, MOSS
*automatically* switches to pure exploitation for that arm without requiring manual
tuning of phase transitions.

**3. Horizon-aware.** Unlike UCB1 (which uses `log t` and never stops exploring),
MOSS uses the known horizon `T` in its bonus. It "knows" that 5,000 rounds is the
budget and sizes exploration accordingly.

**4. Most robust across seeds.** On 100 random seeds MOSS (c=0.16) achieves
mean **520.2 clicks** vs UCB(α=0.02) at 517.2 and decaying ε-greedy at 508.4.
Lower variance (std ≈ 33) means the score on any fixed set of hidden seeds is
likely to be high.

### Why not MOSS (c=0.1)?

MOSS with c=0.10 scores 541.1 on the 10 practice seeds but only 512.4 on 100
seeds (std ≈ 44). It appears to be "lucky" on the specific practice seeds. The
coefficient c=0.16 is more balanced: better on 100 seeds and nearly identical on
the practice seeds (536.6 vs 536.4 for UCB).

### Why not Thompson Sampling?

Thompson Sampling (TS) with Beta(1,1) prior produces good *asymptotic* regret but
explores heavily early on because the posterior variance is high before observations
accumulate. With only 5,000 rounds and 10 arms, TS spends too many impressions on
clearly inferior arms. MOSS explicitly concentrates exploration where it matters
(underpulled arms) without the randomness overhead of sampling.

### Why not KL-UCB?

KL-UCB is theoretically *asymptotically* optimal (matches Lai-Robbins lower bound)
but its `log(t) + c·log(log(t))` bonus over-explores for small horizons like T=5000.
It achieves only 475.3 on practice seeds.

---

## 4. Implementation Details

```python
def select_arm(self):
    self.t += 1
    # Warm-up: pull each arm exactly once
    for a in range(self.n_arms):
        if self.counts[a] == 0:
            return a
    # MOSS index
    log_term = np.maximum(
        np.log(self.horizon / (self.n_arms * self.counts)), 0.0
    )
    ucb = self.values + 0.16 * np.sqrt(log_term / self.counts)
    return int(np.argmax(ucb))
```

**Warm-up phase:** Force one pull of each arm before any MOSS computation. This
avoids division-by-zero and ensures all estimates are initialized. Cost: exactly
K=10 impressions.

**Incremental mean update:** `values[arm] += (reward - values[arm]) / counts[arm]`
is numerically stable and O(1) per step — no risk of overflow.

**Computational complexity:** O(K) per step — well within the evaluation time limit.

**Coefficient tuning:** c=0.16 was selected by grid search over c ∈ {0.05, 0.08,
0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.25, 0.30} evaluated on
100 random seeds to avoid overfitting to the practice seeds.

---

## 5. Exploration vs Exploitation Analysis

With c=0.16 and T=5000, K=10:

- **Early (n_a small):** log(500/n_a) is large → MOSS explores all arms frequently.
- **Mid (n_a ≈ 50):** log(10) ≈ 2.3 → bonus ≈ 0.16·sqrt(2.3/50) ≈ 0.048. For
  arms with CTR difference ≥ 0.05, greedy dominates.
- **Late (n_a ≈ 500):** log(1) = 0 → pure greedy exploitation.

This means ~5,000 impressions are distributed as: ~10 warm-up, ~few hundred for
MOSS exploration, then pure exploitation of the best arm identified so far.

---

## 6. Local Testing Results

```
================================================
  Testing: policy.py
  Practice problem: 10 ads, 5000 impressions, 10 seeds
================================================
  Optimal (always best ad) :    550.0 clicks
  Random baseline          :    336.3 clicks
  >> YOUR POLICY <<        :    536.6 clicks
     regret (lower better) :     18.3
------------------------------------------------
  Excellent — you're hugging the optimal line!
================================================
```

**Gap closed:** (536.6 − 336.3) / (550.0 − 336.3) = **93.8%** of the
random→optimal gap.

**Regret:** 18.3 clicks over 5,000 impressions — less than one suboptimal pull
per 270 impressions on average.

---

## 7. How to Reproduce

```bash
python local_test.py policy.py
```

Expected output: ~536–541 clicks (slight variance due to numpy's RNG seeding order).

---

## 8. References

- Audibert, J.-Y. & Bubeck, S. (2009). *Minimax Policies for Adversarial and
  Stochastic Bandits.* COLT 2009. (MOSS algorithm)
- Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms.* Cambridge University
  Press. (Chapters on UCB and regret bounds)