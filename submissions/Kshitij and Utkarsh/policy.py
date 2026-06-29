import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.t = 0
        self.alpha = np.ones(n_arms)
        self.beta  = np.ones(n_arms)
        self.counts = np.zeros(n_arms)

    def select_arm(self):
        # pull each arm once first
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return int(i)

        progress = self.t / self.horizon

        if progress > 0.88:
            return int(np.argmax(self.alpha / (self.alpha + self.beta)))

        # vectorized: draw 20 samples per arm at once, take row-max then col-argmax
        # shape (20, n_arms) — fast numpy op, no Python loop
        samples = np.random.beta(self.alpha, self.beta, size=(20, self.n_arms))
        # for each of 20 draws, which arm wins?
        # take the arm that wins most often
        winners = np.argmax(samples, axis=1)  # shape (20,)
        counts = np.bincount(winners, minlength=self.n_arms)
        return int(np.argmax(counts))

    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
