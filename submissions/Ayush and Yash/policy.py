# policy.py

import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.t = 0
        
        # We can still use the pessimistic Beta(1, 10) prior we discovered earlier
        # to prevent wasting time on obvious losers, or keep your Beta(1,1). 
        # I'll use 1.0 for both to match your logic.
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        
        self._init_queue = list(range(n_arms))

    def select_arm(self):
        # Phase 1: Pure exploration
        if self._init_queue:
            return self._init_queue.pop(0)

        # Phase 2: Bayes-UCB via Gaussian Approximation
        t = max(self.t, 1)
        
        # 1. Calculate the exact Mean of the Beta distributions
        totals = self.alpha + self.beta
        means = self.alpha / totals
        
        # 2. Calculate the exact Variance of the Beta distributions
        variances = (self.alpha * self.beta) / ((totals ** 2) * (totals + 1.0))
        stdevs = np.sqrt(variances)
        
        # 3. Construct the Quantile Multiplier (c)
        # Instead of scipy's inverse CDF, we use an exploration multiplier 
        # that scales dynamically with the clock, similar to your quantile schedule.
        # As 't' approaches horizon, this shrinks, shifting to pure exploitation.
        c = np.sqrt(np.log(self.horizon / t)) 
        
        # 4. Calculate the optimistic bound: Mean + (c * Standard Deviation)
        ucb_values = means + (c * stdevs)
        
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.t += 1
        if reward > 0:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0