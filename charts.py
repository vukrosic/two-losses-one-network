"""Draws every figure in figures/ from the run data in experiments/*.jsonl and the tables in RESULTS.md.
    python3 charts.py
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, EXP)
os.chdir(EXP)
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

# 6. Hand split: which share of the weights the shared network acts like it gives ---------------------------
share = [10, 20, 30, 50]
dot_err = [19.0, 14.5, 13.2, 10.7]
total = [0.0083, 0.0075, 0.0079, 0.0091]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
a1.plot(share, dot_err, color=ORANGE, lw=2, marker="o", ms=8, mec="white", mew=1.5, label="separate networks, uneven split")
a1.axhline(18.3, color=BLUE, lw=2, ls="--", label="one shared network: 18.3%")
a1.annotate("matches giving it\n~10% of the weights", xy=(11, 18.3), xytext=(23, 16.2), color=INK2, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=INK2))
a1.set(xticks=share, xticklabels=[f"{s}%" for s in share], xlim=(6, 54), ylim=(9, 20.5),
       xlabel="share of the 16,000 weights given to the chasing dot", ylabel="chasing dot's error (%)")
a1.set_title("Shared network ≈ 10% of the weights", loc="left", fontsize=11)
a1.legend(loc="lower left", fontsize=9)
a2.plot(share, [t * 1000 for t in total], color=ORANGE, lw=2, marker="o", ms=8, mec="white", mew=1.5, label="separate networks, uneven split")
a2.axhline(9.6, color=BLUE, lw=2, ls="--", label="one shared network")
a2.annotate("lowest total loss:\n20% to the chasing dot", xy=(20.5, 7.48), xytext=(30, 7.12), color=INK2, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=INK2))
a2.set(xticks=share, xticklabels=[f"{s}%" for s in share], xlim=(6, 54), ylim=(7, 10),
       xlabel="share of the 16,000 weights given to the chasing dot", ylabel="total loss (×1,000)")
a2.set_title("The best trade gives it 20%, twice as much", loc="left", fontsize=11)
a2.legend(loc="upper left", fontsize=9)
fig.tight_layout()
save(fig, "hand_split.png")

# 7. Theory: the shared network keeps the directions worth the most loss --------------------------------------
lam, beta, k = 10.0, 2.0, 20
i = np.arange(1, 31)
vA, vB = lam * i ** -beta, 1.0 * i ** -beta
tau = np.sort(np.r_[vA, vB])[::-1][k - 1]
nA, nB = int((vA >= tau).sum()), int((vB >= tau).sum())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for v, c, name in [(vA, BLUE, "bigger-loss thing"), (vB, ORANGE, "smaller-loss thing")]:
    keep = v >= tau
    a1.scatter(i[keep], v[keep], color=c, s=34, zorder=3, label=f"{name}: kept")
    a1.scatter(i[~keep], v[~keep], facecolors="white", edgecolors=c, s=34, zorder=3, label=f"{name}: dropped")
a1.axhline(tau, color=INK, lw=1.2, ls="--")
a1.text(30.5, tau * 1.25, "keep everything above this line\n(20 directions in total)", ha="right", fontsize=9, color=INK)
a1.set(yscale="log", xlim=(0, 31), xlabel="direction number i (most important first)",
       ylabel="loss explained by direction i")
a1.set_title(f"Shared rank-20 network, Λ = 10, β = 2", loc="left", fontsize=11)
a1.legend(fontsize=8.5, loc="upper right")
a2.bar([0, 1], [nA, k // 2], color=BLUE, width=0.6, edgecolor="white", linewidth=2, label="bigger-loss thing")
a2.bar([0, 1], [nB, k // 2], bottom=[nA, k // 2], color=ORANGE, width=0.6, edgecolor="white", linewidth=2, label="smaller-loss thing")
for x, (a, b) in enumerate([(nA, nB), (k // 2, k // 2)]):
    a2.text(x, a / 2, str(a), ha="center", va="center", color="white", fontweight="bold")
    a2.text(x, a + b / 2, str(b), ha="center", va="center", color="white", fontweight="bold")
a2.set(xticks=[0, 1], xticklabels=["one shared\nnetwork", "one network\nper thing"], ylim=(0, 26), yticks=[0, 5, 10, 15, 20],
       ylabel="directions kept")
a2.legend(fontsize=8.5, loc="upper center", ncol=2)
a2.grid(axis="x", visible=False)
a2.set_title("Directions each thing gets", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "directions.png")
print(f"directions: shared keeps {nA} + {nB}")

# 8. Learning curves: the shared network is the own network, slowed by a fixed factor ------------------------
R = [json.loads(l) for l in open("drive.jsonl")]
def base_ok(r):
    return ("carscale" in r and r.get("decay") and r.get("world", "drive") == "drive" and r.get("loss") != "l1" and r.get("norm") == "none"
            and r.get("lr") == 1e-3 and r.get("batch") == 256 and r.get("depth", 2) == 2 and r.get("nclip", 1000) == 1000 and r["arch"] == "mlp"
            and r.get("opt", "adam") == "adam" and r.get("lw") in (None, [1.0, 1.0]) and r.get("tag") in ("steps", "", None, "fine")
            and r["size"] == "m" and r.get("scalething", 0) == 0)
own = sorted({r["steps"]: r["cv"][1] for r in R if base_ok(r) and r["net"] == "split"}.items())
os_, oe = np.log([s for s, _ in own]), np.log([e for _, e in own])
own_at = lambda S: np.exp(np.interp(np.log(S), os_, oe))
steps_at = lambda E: np.exp(np.interp(-np.log(E), -oe, os_))
fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot([s for s, _ in own], [e for _, e in own], color=INK2, lw=2, marker="o", ms=6, mec="white", label="pedestrian's own network")
for a, c, g in [(1.0, AQUA, 5.2), (4.0, BLUE, 21)]:
    sh = sorted({r["steps"]: r["cv"][1] for r in R if base_ok(r) and r["net"] == "single" and r["carscale"] == a}.items())
    F = 6000 / steps_at(dict(sh)[6000])
    S = np.logspace(np.log10(own[0][0] * F), np.log10(24000), 50)
    ax.plot(S, own_at(S / F), color=c, lw=1.5, ls=":")
    ax.plot([s for s, _ in sh], [e for _, e in sh], color=c, lw=0, marker="s", ms=8, mec="white", mew=1.2,
            label=f"shared network, car moves {g}× farther (F = {F:.1f})")
    print(f"curves: gap {g} F {F:.2f}")
ax.set(xscale="log", yscale="log", xlabel="training steps (each point a full run, learning rate decayed to zero)",
       ylabel="pedestrian's error (÷ keep-velocity guess)")
ax.set_title("Shared network = own network slowed by a fixed factor F", loc="left", fontsize=11.5)
ax.plot([], [], color=INK2, lw=1.5, ls=":", label="own curve stretched by F along the step axis")
ax.set_yticks([0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
ax.set_xticks([250, 1000, 3000, 6000, 12000, 24000]); ax.set_xticklabels(["250", "1k", "3k", "6k", "12k", "24k"]); ax.minorticks_off()
ax.legend(fontsize=9, loc="lower left")
save(fig, "curves.png")

# 9. The fix: divide each thing's loss by its own size --------------------------------------------------------
def gridP(norm, tags):
    out = {}
    for size in ["xs", "s", "sm", "m", "ml"]:
        rs = [r for r in R if r.get("norm") == norm and r.get("tag") in tags and r["size"] == size and r.get("steps") == 6000
              and r.get("decay") and r.get("lr") == 1e-3 and r.get("batch") == 256 and r.get("depth", 2) == 2 and r["arch"] == "mlp"
              and r.get("loss") == "l2" and r.get("nclip", 1000) == 1000 and r.get("opt", "adam") == "adam"
              and r.get("world", "drive") == "drive" and r.get("scalething", 0) == 0 and r.get("peds") == [0.006]
              and r.get("lw") in (None, [1.0, 1.0])]
        s = [r for r in rs if r["net"] == "split" and r["carscale"] == 1][-1]
        pts = sorted({r["carscale"]: r["cv"][1] / s["cv"][1] for r in rs if r["net"] == "single" and r["carscale"] in (0.5, 1, 2, 4, 8)}.items())
        out[(size, s["params"])] = pts
    return out
RAMP = ["#b7d3f2", "#86b6ef", "#2a78d6", "#1a5aa8", "#0d2f5e"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
for ax, (norm, tags, title) in zip(axs, [("none", (None, "fine"), "Plain sum of the two losses"),
                                         ("agent", ("agentnorm",), "Each loss divided by its own size")]):
    for ((size, n), pts), c in zip(gridP(norm, tags).items(), RAMP):
        ax.plot([a * 5.2 for a, _ in pts], [p for _, p in pts], color=c, lw=2, marker="o", ms=6, mec="white", label=f"{n:,} weights")
    ax.axhline(1, color=INK2, lw=1)
    ax.set(xscale="log", yscale="log", xticks=[2.6, 5.2, 10.4, 21, 42], xticklabels=["2.6", "5.2", "10", "21", "42"],
           ylim=(0.6, 35), xlabel="gap: how many times farther the car moves per step")
    ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=11.5)
axs[0].set_yticks([0.7, 1, 2, 5, 10, 30]); axs[0].set_yticklabels(["0.7", "1", "2", "5", "10", "30"])
axs[0].set_ylabel("P = pedestrian's error, shared ÷ own")
axs[0].legend(fontsize=8.5, title="network size", title_fontsize=9)
fig.tight_layout()
save(fig, "fix.png")

# 10. A constant learning rate exaggerates the harm ----------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
lab = ["constant\nlearning rate", "learning rate\ndecays to zero", "4× bigger\nbatch", "average of the\nlast quarter"]
vals = [1.81, 1.30, 1.19, 1.23]
a1.bar(range(4), vals, color=["#9b9a94", BLUE, BLUE, BLUE], width=0.62, edgecolor="white", linewidth=2)
for x, v in enumerate(vals):
    a1.text(x, v + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=10)
a1.set(xticks=range(4), ylim=(1, 2), ylabel="P, pedestrian (driving, same 6,000 steps)")
a1.set_xticklabels(lab, fontsize=8.5)
a1.grid(axis="x", visible=False)
a1.set_title("Any way of removing the jitter lowers P", loc="left", fontsize=11)
worlds = ["Dot world,\nchasing dot", "Arm,\nmedium speed", "Arm,\nfast", "Driving,\n8k weights"]
const, dec = [1.9, 2.4, 4.1, 1.8], [2.0, 1.06, 1.34, 1.41]
x = np.arange(4)
a2.bar(x - 0.18, const, width=0.34, color="#9b9a94", edgecolor="white", linewidth=1.5, label="constant learning rate")
a2.bar(x + 0.18, dec, width=0.34, color=BLUE, edgecolor="white", linewidth=1.5, label="decayed to zero")
for xi, (c, d) in enumerate(zip(const, dec)):
    a2.text(xi - 0.18, c + 0.05, f"{c:.1f}", ha="center", fontsize=9.5, color=INK)
    a2.text(xi + 0.18, d + 0.05, f"{d:.2f}".rstrip("0") if d < 2 else f"{d:.1f}", ha="center", fontsize=9.5, color=INK)
a2.set(xticks=x, ylim=(0, 4.6), ylabel="P, smaller-loss thing")
a2.set_xticklabels(worlds, fontsize=8.5)
a2.grid(axis="x", visible=False)
a2.legend(fontsize=9, loc="upper left")
a2.set_title("Up to 3× smaller once the rate decays", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "const_lr.png")
