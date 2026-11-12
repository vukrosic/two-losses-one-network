"""Draws every figure in figures/ from the run data in toy/*.jsonl and the tables in RESULTS.md.
    python3 charts.py
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOY = os.path.join(HERE, "toy")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TOY)
os.chdir(TOY)
import allfit  # noqa: E402  (reads drive.jsonl from the cwd)

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 100})
law = lambda L: 0.15 * np.asarray(L, float) ** 0.75


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote figures/" + name)


# 1. Every shared network against the loss ratio ------------------------------------------------------------
pts = {"Driving: car and pedestrians": [], "Robot arm: slow base, fast arm": [], "Two dots seen in pictures": [],
       "Teacher networks: no physics": []}
for t in ["base:1", "fine:1", "long18k:1", "weave:1", "twoped:1", "twoped:2", "l1:1", "lr3e-3:1", "lr3e-4:1",
          "batch64:1", "clip250:1", "ped12:1"]:
    pts["Driving: car and pedestrians"] += [(p["lam"], p["P1"]) for p in allfit.points(t)]
for t in ["arm:0", "armfast:0"]:
    pts["Robot arm: slow base, fast arm"] += [(p["lam"], p["P1"]) for p in allfit.points(t)]
T = [json.loads(l) for l in open("teacher.jsonl")]
TT = [r for r in T if r["tag"] in ("grid", "opt") and r["steps"] == 3000 and r["opt"] == "adam"]
key = lambda r: (r["tag"], r["hs"], r["snoise"], r["hf"], r["depth"], r.get("sep", False), r.get("ntrain", 0), r["w"])
sp = {key(r): r for r in TT if r["net"] == "split"}
for r in TT:
    if r["net"] == "single" and key(r) in sp:
        s = sp[key(r)]
        lam = r["a"] ** 2 * s["cv"][0] * s["var"][0] / (s["cv"][1] * s["var"][1])
        pts["Teacher networks: no physics"].append((lam, r["cv"][1] / s["cv"][1] - 1))
D = [json.loads(l) for l in open("two_vs_one.jsonl")]
D = [r for r in D if "latscale" in r and r.get("decay") and r["steps"] == 6000 and r.get("latthing", 0) == 0
     and r.get("opt", "adam") == "adam"]
dsp = {r["widths"][0]: r for r in D if r["net"] == "split" and r["latscale"] == 1}
for r in D:
    if r["net"] == "single" and r["widths"][0] in dsp:
        s = dsp[r["widths"][0]]
        lam = s["test"][0] * r["ch_robot"] ** 2 / (s["test"][1] * r["ch_stranger"] ** 2)
        pts["Two dots seen in pictures"].append((lam, r["test"][1] / s["test"][1] - 1))
for k in pts:
    pts[k] = [(l, p) for l, p in pts[k] if p > 0.05 and l > 0]
allp = [x for v in pts.values() for x in v]
inside = sum(abs(math.log(p / law(l))) < math.log(2.5) for l, p in allp)
print(f"law figure: {len(allp)} shared networks, {inside} within x2.5 of 0.15 Lambda^0.75")
for k, v in pts.items():
    print(f"  {k:32s} n {len(v):3d}  median measured/law x{math.exp(np.median([math.log(p / law(l)) for l, p in v])):.2f}")

fig, ax = plt.subplots(figsize=(7.6, 5.6))
L = np.logspace(-1, 6, 100)
ax.fill_between(L, law(L) / 2.5, law(L) * 2.5, color="#9b9a94", alpha=0.16, lw=0, label="within ×2.5 of the line")
for (k, v), c in zip(pts.items(), [BLUE, ORANGE, AQUA, YELLOW]):
    ax.scatter(*zip(*v), s=16, color=c, alpha=0.7, edgecolors="white", linewidths=0.4, label=f"{k} ({len(v)})")
ax.plot(L, law(L), color=INK, lw=2)
ax.text(2e4, law(2e4) / 6, "P − 1 = 0.15 Λ$^{0.75}$", color=INK, fontsize=12, fontweight="bold")
ax.set(xscale="log", yscale="log", xlim=(0.1, 1e6), ylim=(0.05, 1500),
       xlabel="Λ = bigger loss ÷ smaller loss (each measured with its own network)",
       ylabel="P − 1 = extra error of the smaller-loss thing")
ax.set_title(f"One line fits {len(allp)} shared networks in four different worlds", loc="left")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "law.png")

# 2. Training longer does not shrink the penalty -------------------------------------------------------------
steps = [6000, 12000, 24000]
series = [("Driving, car 21× the pedestrian's move", [3.39, 3.04, 3.08], BLUE, "-"),
          ("Driving, car 10× the pedestrian's move", [1.81, 1.81, 1.72], BLUE, "--"),
          ("Driving, car 5× the pedestrian's move", [1.30, 1.33, 1.41], BLUE, ":"),
          ("Two dots, robot target scaled ×4", [3.78, 3.82, 3.70], AQUA, "-"),
          ("Robot arm, base at its error floor", [2.75, 2.20, 1.67], ORANGE, "-")]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
nudge = {"Driving, car 10× the pedestrian's move": 0.07, "Robot arm, base at its error floor": -0.07}
for name, ys, c, ls in series:
    ax.plot(steps, ys, color=c, ls=ls, lw=2, marker="o", ms=7, mec="white", mew=1.5)
    ax.text(25500, ys[-1] + nudge.get(name, 0), name, color=INK2, va="center", fontsize=9.5)
ax.set(xscale="log", xticks=steps, xticklabels=["6,000", "12,000", "24,000"], xlim=(5200, 26000), ylim=(1, 4.2),
       xlabel="training steps (learning rate decayed to zero in each run)",
       ylabel="P = smaller thing's error, shared ÷ own")
ax.minorticks_off()
ax.set_title("4× more training leaves the penalty unchanged while the smaller thing is still learning", loc="left", fontsize=11.5)
save(fig, "steps.png")

# 3. Size helps only when the separate networks have stopped gaining from size --------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
for st, c in [(300, ORANGE), (3000, BLUE)]:
    rs = [r for r in T if r["tag"] == "short" and r["steps"] == st]
    sps = {r["w"]: r for r in rs if r["net"] == "split"}
    ws = sorted(sps)
    N = [sps[w]["N"] for w in ws]
    own = [sps[w]["cv"][1] for w in ws]
    P = [[r for r in rs if r["net"] == "single" and r["w"] == w and r["a"] == 8][-1]["cv"][1] / sps[w]["cv"][1] for w in ws]
    lab = f"{st:,} steps"
    a1.plot(N, own, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=lab)
    a2.plot(N, P, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=lab)
a1.set(xscale="log", yscale="log", xlabel="total weights", ylabel="smaller-loss target's error, own network")
a1.set_yticks([0.06, 0.1, 0.2, 0.3, 0.4]); a1.set_yticklabels(["0.06", "0.1", "0.2", "0.3", "0.4"]); a1.minorticks_off()
a1.set_title("Own network: at 300 steps more weights barely help", loc="left", fontsize=11)
a2.set(xscale="log", xlabel="total weights", ylabel="P = error, shared ÷ own (target ×8)", ylim=(1, 4.4))
a2.set_title("Shared network: size lowers P only at 300 steps", loc="left", fontsize=11)
a1.legend(); a2.legend()
fig.tight_layout()
save(fig, "size.png")

# 4. A loss weight and a change of units are the same knob ---------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for name, w, pw, u, pu, c in [("Driving", [0.25, 4, 16, 64], [1.06, 2.22, 4.25, 7.03], [0.25, 4, 16, 64], [1.18, 1.81, 3.39, 6.49], BLUE),
                              ("Two dots", [0.25, 4, 16], [1.30, 2.29, 3.66], [0.25, 4, 16], [1.31, 2.43, 3.78], AQUA)]:
    ax.plot(w, pw, color=c, lw=2, marker="o", ms=8, mec="white", mew=1.5, label=f"{name}: bigger thing's loss × a²")
    ax.plot(u, pu, color=c, lw=0, marker="s", ms=8, mfc="white", mec=c, mew=2, label=f"{name}: bigger thing measured in units × a")
ax.set(xscale="log", xticks=[0.25, 1, 4, 16, 64], xticklabels=["¼", "1", "4", "16", "64"], ylim=(0.8, 7.5),
       xlabel="a²  (loss weight, or the square of the unit change)", ylabel="P = smaller thing's error, shared ÷ own")
ax.minorticks_off()
ax.set_title("Weighting a loss by a² does what measuring in a× bigger units does", loc="left", fontsize=12)
ax.legend(fontsize=9.5)
save(fig, "weight_units.png")

# 5. Theory against the measured law -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))
L = np.logspace(0, 3, 100)
for b, c in [(2, "#86b6ef"), (3, BLUE), (4, "#104281")]:
    ax.plot(L, ((1 + L ** (1 / b)) / 2) ** (b - 1), color=c, lw=2)
    ax.text(1080, ((1 + 1000 ** (1 / b)) / 2) ** (b - 1), f"theory, β = {b}", color=INK2, va="center", fontsize=9.5)
ax.plot(L, 1 + law(L), color=ORANGE, lw=2.5, ls="--")
ax.text(1080, 1 + law(1000) * 0.82, "measured law", color=ORANGE, va="center", fontsize=9.5, fontweight="bold")
ax.set(xscale="log", yscale="log", xlim=(1, 1000), ylim=(1, 45), xlabel="Λ = bigger loss ÷ smaller loss",
       ylabel="P = smaller thing's error, shared ÷ own")
ax.set_yticks([1, 2, 5, 10, 20, 40]); ax.set_yticklabels(["1", "2", "5", "10", "20", "40"])
ax.set_title("A shared linear network gives the same curve, with no training", loc="left", fontsize=12)
save(fig, "theory.png")
