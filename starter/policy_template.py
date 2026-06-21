# policy.py  --  YOUR submission goes in a class named exactly `Policy`.

class Policy:
    """Your ad-recommendation policy.

    The evaluator does this:
        policy = Policy(n_arms, horizon)
        for t in range(horizon):
            arm = policy.select_arm()        # which ad do you show?
            reward = show_ad(arm)            # 1 = click, 0 = no click
            policy.update(arm, reward)       # learn from it
    ...then adds up your clicks. Maximise total clicks!
    """

    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms        # number of ads to choose from
        self.horizon = horizon      # total number of impressions you get
        # TODO: remember whatever you need (counts, running estimates, ...)

    def select_arm(self):
        # TODO: return the index of the ad to show this round (0 .. n_arms-1)
        raise NotImplementedError

    def update(self, arm, reward):
        # TODO: update your beliefs using the reward you just saw (0 or 1)
        raise NotImplementedError
