import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.counts = [0] * n_arms
        self.rewards = [0.0] * n_arms
        self.t = 0  # current timestep

    def select_arm(self):
        self.t += 1
        #commit after _% exploration
        if self.t > 0.4 * self.horizon:
            return np.argmax(self.rewards)
        #explore before _%
        elif self.t < 0.2 * self.horizon:
            epsilon = 1.0/self.t**0.2
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.n_arms)
            else:
                return np.argmax(self.rewards)
        else:
            epsilon = 0.5/self.t**0.3
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.n_arms)
            else:
                return np.argmax(self.rewards)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += (reward - self.rewards[arm]) / self.counts[arm]

#387.9 ,Rank 5 , 37.9 with 30 (pow=0.2) else pow-0.5 then commit after 60-70%