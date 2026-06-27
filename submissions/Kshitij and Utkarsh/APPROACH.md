# Approach: Voting Thompson Sampling with Late Greedy Switch

## Algorithm

We implement **Voting Thompson Sampling** — a variant of Thompson Sampling designed to handle arms with very close click-through rates (CTRs).

At each step:
1. Draw 20 independent sample sets from all arms' Beta posteriors simultaneously (vectorized, shape 20×10)
2. For each of the 20 draws, find the winning arm (argmax)
3. Pick the arm that wins the most votes across all 20 draws
4. After 88% of horizon exhausted: switch to pure greedy (argmax of posterior mean)
5. Warmup: pull each arm exactly once before any exploitation begins

## Why This Algorithm

**Problem with vanilla Thompson Sampling:** A single Beta draw is noisy. When two arms have CTRs 0.07 and 0.075 (very close), a single draw frequently picks the wrong arm just by chance.

**Why voting fixes it:** Drawing 20 samples and taking the majority vote approximates the arm with the highest posterior *probability of being best*:

```
P(arm i is best) ≈ fraction of draws where arm i wins
```

The arm with genuinely higher CTR will win more draws even when CTRs are close. This directly targets the problem structure — "top few ads have very close CTRs."

**Mathematical intuition:**
- After n pulls, posterior mean ≈ true CTR, variance ≈ p(1-p)/n
- With CTR gap Δ ≈ 0.01, voting helps identify winner earlier by aggregating information across 20 independent draws
- Late greedy switch at t=0.88×H ensures last ~600 impressions all go to best-known arm

## Design Decisions

| Choice | Value | Reason |
|--------|-------|--------|
| Prior | Beta(1,1) | Uninformative — let data speak, CTRs are unknown |
| Vote draws | 20 | Balances discrimination quality vs speed (fully vectorized) |
| Warmup | 1 pull per arm | Just enough to initialize counts, avoids cold start |
| Greedy switch | 88% horizon | Last ~600 steps pure exploit, no wasted exploration |

## Alternatives Considered

### 1. Vanilla Thompson Sampling (single draw)
- **Pro:** Simple, theoretically sound
- **Con:** Single draw too noisy for close CTRs

### 2. UCB1 Pure
- **Pro:** Logarithmic regret guarantee, deterministic
- **Con:** Exploration bonus keeps visiting bad arms too long at low CTRs

### 3. Hybrid UCB + Thompson + Mean blend
- **Pro:** Three signals combined
- **Con:** Too many hyperparams, sensitive to tuning, performed worse on real hidden CTRs

### 4. Warmup 20 pulls per arm
- **Pro:** Strong initial estimates per arm
- **Con:** 200 forced impressions on bad arms too costly at low CTR regime

### 5. Epsilon-greedy with decay
- **Con:** Explores uniformly regardless of posterior — ignores accumulated belief structure

## Expected Performance

Voting TS handles close CTRs well because:
- More draws = better approximation of P(arm i is best)
- Naturally concentrates on top arms faster than single draw
- Fully vectorized — no timeout risk
- Failure mode: bad luck in early pulls can delay convergence — mitigated by Beta posterior recovering over time as evidence accumulates
