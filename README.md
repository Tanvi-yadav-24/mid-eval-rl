# 💥 Mid-Eval: Reinforcements Are In — Submissions

Welcome to the **ad-recommender bandit challenge**. You run an ad slot with
**10 ads**, you get **10,000 impressions**, and each ad has a *hidden, fixed*
click-through-rate. Build a policy that figures out which ads people click and
racks up the most clicks. Teams of **2**.

> 🎮 **Practice arena (live leaderboard):** <YOUR-RAILWAY-URL>
> Upload your `policy.py` there as often as you like to test it and climb the board.
>
> 📦 **This repo is for your FINAL graded submission**, via Pull Request.

---

## 🚀 Quick start

1. Read the **[problem statement](problem_statement.md)**.
2. Copy **[`starter/policy_template.py`](starter/policy_template.py)** → `policy.py` and fill in the two TODOs.
   Or start from the worked **[`starter/thompson_sampling.py`](starter/thompson_sampling.py)**.
3. Test it offline:
   ```bash
   cd starter
   python local_test.py ../path/to/your/policy.py
   ```
4. Upload it on the website to see your real score and rank.
5. When you're happy, **submit via Pull Request** (below).
6. Want ideas to go beyond Thompson Sampling? See **[HINTS.md](HINTS.md)**.

You only need **Python 3** and **numpy** (`pip install numpy`).

---

## 📦 How to submit (Pull Request)

1. **Fork** this repository (top-right "Fork" button) and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mid-eval-rl.git
   cd mid-eval-rl
   ```
2. Inside **`submissions/`**, make **one folder** named with **both members' first
   names**, joined by `" and "`. Example:
   ```
   submissions/Archit and Utkarsh/
   ```
3. Put exactly **two files** in your folder:

   | File | What it is |
   |------|------------|
   | `policy.py` | Your final policy (the `Policy` class — same format as the template). |
   | `APPROACH.md` | Your write-up: the algorithm, **why** you chose it, what else you tried, and your results. A PDF is fine too. |

   See **[`submissions/Archit and Utkarsh/`](submissions/Archit%20and%20Utkarsh)** for a complete example.
4. Commit and push to your fork:
   ```bash
   git add "submissions/Your Name and Their Name"
   git commit -m "Submission: <Your Team Name>"
   git push origin main
   ```
5. Open a **Pull Request** into this repo. Title it `Submission: <Your Team Name>`
   and fill in the checklist that appears.

### ✅ Submission rules
- Folder name = **first names of both members**, e.g. `Archit and Utkarsh`.
- Exactly one `policy.py` + one write-up (`APPROACH.md` or PDF). Nothing else.
- Only edit files **inside your own folder**. Don't touch other teams' folders or the starter kit.
- `policy.py` must define a class named exactly **`Policy`** with `__init__(self, n_arms, horizon)`, `select_arm(self)`, and `update(self, arm, reward)`.
- Only `numpy` + the Python standard library. No internet, no file I/O, no extra packages.
- No hard-coding ad indices or click-rates — they're hidden and change per seed. Your policy must **learn**.

**Deadline:** **Sunday, 28 June 2026 (end of day).**

Good luck — may the best bandit win! 🏆
