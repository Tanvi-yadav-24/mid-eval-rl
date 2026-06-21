# 🧠 Hints, Clues & Questions

You already know **ε-greedy**, **explore-then-commit (ETC)**, and **Thompson
Sampling**. All three work on this problem — but the top ads have *very close*
click-rates, so lazy exploration gets punished. Here's how to push past the
baseline.

## Directions to explore

- **Optimism in the face of uncertainty (UCB).** Instead of picking the ad with
  the best *average* so far, pick the one with the best *plausible* value: its
  average **plus** a bonus that's large when you've tried it few times and shrinks
  as evidence piles up. Look up **UCB1** — it's a few lines of code.

- **Decay your exploration.** A fixed ε explores forever and keeps paying for it.
  An ε that **shrinks over time** (e.g. proportional to `1/t` or `sqrt(log t / t)`)
  explores hard early, then cashes in. Can you derive a good schedule?

- **Pick ETC's exploration length on purpose.** Explore-then-commit lives or dies
  by how many rounds you explore before locking in. Too short → you crown the wrong
  ad. Too long → you waste clicks. Can you choose the length from `n_arms` and
  `horizon` instead of guessing?

- **Tune Thompson's prior.** `Beta(1, 1)` is uniform. What happens with a more
  optimistic prior, or one that encodes "most ads are bad"? Small changes can move
  your score.

- **Use the clock.** You *know* the horizon. Early impressions are for learning,
  late impressions are for earning. Should your behaviour change as `t` approaches
  the horizon?

## Questions to guide you

1. When are two ads "too close to call"? Roughly how many times must you show each
   before you can trust which is better? (Think about the variance of an average of
   coin flips.)
2. If you commit to the wrong ad early, how many clicks does that cost you over
   10,000 impressions? (That gap *is* regret.)
3. Why do we average over **30 seeds**? How could you do the same when tuning
   locally so you're measuring skill, not luck? (Hint: `local_test.py` already does.)
4. Random does ~50% CTR-weighted clicks; optimal hugs the best ad. **What fraction
   of that gap** does your policy close — and which idea moves it the most?

## A good workflow
1. Start from `starter/thompson_sampling.py`, run `local_test.py`, note the score.
2. Change **one idea at a time** (e.g. add a UCB bonus). Re-run `local_test.py`.
3. Keep what helps. When you've got something strong, upload it to the website to
   see your real rank.
4. Write down *what you tried and why* as you go — that's literally your `APPROACH.md`.

> Remember: the practice CTRs in `local_test.py` are **not** the real hidden ones.
> Don't hard-code anything — build something that genuinely learns.
