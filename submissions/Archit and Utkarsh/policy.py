# policy.py  --  EXAMPLE submission (decaying epsilon-greedy).
# This is here to show the expected FORMAT and quality. Build your own!

import numpy as np


class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.counts = np.zeros(n_arms)      # times each ad was shown
        self.values = np.zeros(n_arms)      # running average reward per ad
        self.t = 0

    def select_arm(self):
        self.t += 1

        # 1) Make sure every ad is tried at least once.
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        # 2) Decaying epsilon: explore a lot early, then mostly exploit.
        epsilon = min(1.0, self.n_arms / self.t)
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_arms))   # explore
        return int(np.argmax(self.values))               # exploit best so far

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean: values[arm] += (reward - values[arm]) / n
        self.values[arm] += (reward - self.values[arm]) / n
