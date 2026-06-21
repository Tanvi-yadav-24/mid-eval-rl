# policy.py  --  Random baseline. This is the floor: it ignores all feedback.
# If your clever algorithm can't beat this, something is wrong!

import random

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms

    def select_arm(self):
        return random.randrange(self.n_arms)

    def update(self, arm, reward):
        pass
