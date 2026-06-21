# policy.py  --  Thompson Sampling for a Bernoulli bandit (WORKED EXAMPLE).
#
# Idea: keep a Beta(alpha, beta) belief over each ad's true click-rate. Each
# round, draw one sample from every ad's belief and show the ad with the highest
# sample. Ads we're unsure about occasionally draw high -> we keep exploring them;
# ads that keep failing get sampled low -> we stop wasting impressions on them.

import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.alpha = np.ones(n_arms)   # 1 + clicks seen on each ad
        self.beta = np.ones(n_arms)    # 1 + no-clicks seen on each ad

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
