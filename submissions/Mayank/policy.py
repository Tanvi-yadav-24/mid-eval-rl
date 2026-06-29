import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.counts = [0] * n_arms
        self.values = [0] * n_arms
        self.t = 0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            avg_reward = self.values[arm] / self.counts[arm]
            bonus = np.sqrt(0.005 * (np.log(self.t)) / self.counts[arm])
            ucb_values[arm] = avg_reward + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        self.values[arm] += reward
