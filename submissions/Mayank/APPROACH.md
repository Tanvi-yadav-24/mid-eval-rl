# **Approach — Team Clickbaiter**

## **1\. Understanding the Problem**

I am given an ad slot. I have to pick one of the 10 ads(arms), each with their own hidden ctr to display every time a user shows up. I get 5000 impressions to learn about the ads and maximize the ctr I get. So, I need to make a bot that finds the best ad while minimizing the exploration.

---

## **2\. What We Chose and Why**

We used **UCB (Upper Confidence Bound) with a tuned exploration constant**:

UCB score \= empirical\_CTR \+ sqrt(0.005 \* ln(total\_pulls) / pulls\_on\_this\_arm)

### **Why UCB is ideal for tuning and similar problems:**

The real power of UCB is **reproducibility and generalizability**:

1. **Deterministic → Tunable**: UCB always produces the same arm selection for identical data states. This means:

   * We can run experiments, observe the constant's effect, and tune precisely  
   * No randomness confounding the results — if constant 0.005 works, it works  
   * Easy to diagnose why a constant is too high/low  
2. **Problem-specific optimization**: Once we tune the constant for a problem type (low values of arms), we have a formula that *transfers* to similar problems:

   * If you face another bandit problem with similar structure, constant 0.005 is a great starting point  
   * Thompson Sampling's optimal prior depends on the specific hidden CTRs, no transfer learning

This contrasts with Thompson Sampling, where tuning the prior is opaque: "should alpha=0.5 and beta=1.5?". With UCB, the knob is clear and interpretable.

---

## **3\. What Else I Tried**

| Approach | Local avg clicks | Notes |
| ----- | ----- | ----- |
| Thompson Sampling (Beta priors) | 477.3 | Sensitive to prior choice; beta(1,1) underperforms |
| Thompson Sampling (tuned priors) | 523.2 | with (1,36), seemed to perform well in local tests but not so good in online test cases and not very predictable and easy to changes |
| UCB (constant \= 2\) | 382.9 | Bonus is too large compared to the values till the end, ends up with too much exploration. |
| UCB (constant \= 0.5) | 439.8 | Better than before, but could be improved |
| UCB (constant \= 0.05) | 511.4 | Decent but leaves clicks on table |
| **UCB (constant=0.005)** | **535.2** | **closed 93% of the gap** |

---

## **4\. Results**

**Local practice score** (on PRACTICE\_CTRS over 10 seeds):

* Optimal (always best ad): 550 clicks  
* Random baseline: 336.3 clicks  
* **My UCB policy: 535.2 clicks**  
* **Gap closed: (535.2 \- 336.3) / (550 \- 336.3) \= 93%**  
* Regret: \~minimal, concentrated in early rounds

---

## **5\. What We'd Do Next**

1. **Adaptive constant**: Instead of fixed 0.005, use `0.005 * sqrt(ln(t) / (t+1))` to decay over time.  
2. **Forced batch exploration**: Every N rounds, guarantee a round-robin pull of each arm to keep estimates fresh.

---

## **How to Reproduce**
```bash
cd starter
python local_test.py "../submissions/Mayank/policy.py"
```

