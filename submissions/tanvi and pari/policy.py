# policy.py -- UCB1 (tuned exploration constant) for a Bernoulli bandit.
#
# Idea: instead of picking the ad with the best average so far (greedy, risks
# locking onto the wrong ad) or exploring uniformly at random (epsilon-greedy,
# wastes pulls on ads we're already sure are bad), pick the ad with the best
# *plausible* value: its running mean reward plus a confidence bonus that's
# large when we've barely tried it and shrinks as evidence piles up. This is
# "optimism in the face of uncertainty" (UCB1). Exploration is automatically
# targeted at the ads we're least sure about -- exactly where this problem is
# hard, since the top few ads have very close click-rates.

import numpy as np


class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.counts = np.zeros(n_arms)   # times each ad was shown
        self.values = np.zeros(n_arms)   # running average reward per ad
        self.t = 0
        # Exploration strength. The textbook UCB1 bonus uses sqrt(2 * ln(t)/n),
        # i.e. c = sqrt(2) ~= 1.41, which is provably safe for *any* horizon.
        # Because we know the exact horizon (5000) and only have 10 arms, that
        # bonus is far larger than we need -- it keeps re-checking ads long
        # after they're clearly worse. We tuned c down empirically on the
        # practice CTRs (see APPROACH.md) and found a much smaller bonus
        # closes more of the random-to-optimal gap on this specific horizon.
        self.c = 0.15

    def select_arm(self):
        self.t += 1

        # Force one pull of every ad up front so no estimate starts blind.
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean: values[arm] += (reward - values[arm]) / n
        self.values[arm] += (reward - self.values[arm]) / n
