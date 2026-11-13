"""Slowdown factor F per shared run: the slower thing's shared error after S steps equals its own-network error after S/F steps.
Own curves: split runs of the same scenario at other step counts (tag CURVE[scenario]) plus the scenario's own split run.
    python3 fslow.py            pooled fit F - 1 = c (Lam/N)^b and per-scenario offsets"""
import json, numpy as np, allfit
R = [json.loads(l) for l in open("drive.jsonl")]
CURVE = {"base": "steps", "long18k": "steps", "pedfast": "steps", "fine": "steps", "arm": "arm-steps", "armfast": None,
         "weave": "weave-steps", "muon": "muon-steps", "grubig": "steps"}
ARCH = {"grubig": "gru"}
def curve(scen, size, slow, ref):
    tags = {scen, CURVE[scen]}
    keys = ("world", "arch", "lr", "batch", "depth", "nclip", "loss", "norm", "curve", "armgain", "peds")
    D = dict(world="drive", lr=0.001, batch=256, depth=2, loss="l2", norm="none", curve=0.0, armgain=0.06, peds=[0.006], decay=True, opt="adam")
    g = lambda r, k: r.get(k) if r.get(k) is not None else D.get(k)
    d = {}
    for r in R:
        if r["net"] == "split" and r.get("cv") and r["size"] == size and (r.get("tag") or "base") in tags and g(r, "decay") and all(g(r, k) == g(ref, k) for k in keys + ("opt",)):
            d[r["steps"]] = r["cv"][slow]
    return sorted(d.items())
def seff(c, E):
    s = np.log([x for x, _ in c]); e = np.log([y for _, y in c])
    if not (e.min() <= np.log(E) <= e.max()): return np.nan
    return float(np.exp(np.interp(-np.log(E), -e, s)))
pts = []
for scen in [k for k, v in CURVE.items() if v]:
    slow = 0 if scen in ("pedfast", "arm", "armfast") else 1
    allfit.KEEPL = scen == "grubig"
    for p in allfit.points("%s:%d" % (scen, slow)):
        ref = next(r for r in reversed(R) if r.get("carscale") is not None and (r.get("tag") or "base") == scen and r["net"] == "split" and r["size"] == p["size"])
        c = curve(scen, p["size"], slow, ref)
        if len(c) < 4: continue
        se = seff(c, (p["P1"] + 1) * dict(c)[p["steps"]])
        if np.isfinite(se): pts.append(dict(t=scen, size=p["size"], lam=p["lam"], N=p["N"], F=p["steps"] / se, P=p["P1"] + 1, r=p["r"]))
q = [p for p in pts if p["F"] > 1.05]
y = np.log([p["F"] - 1 for p in q]); x = np.log([p["lam"] / p["N"] for p in q])
b, a = np.polyfit(x, y, 1); res = y - (a + b * x)
print("F - 1 = %.1f * (Lam/N)^%.2f   R2 %.3f  rms x%.2f  n %d" % (np.exp(a), b, 1 - res.var() / y.var(), np.exp(res.std()), len(q)))
X = np.c_[np.ones(len(q)), np.log([p["lam"] for p in q]), np.log([p["N"] for p in q])]
cc, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - (y - X @ cc).var() / y.var()
print("free: F - 1 ~ Lam^%.2f N^%.2f  R2 %.3f" % (cc[1], cc[2], r2))
for t in sorted({p["t"] for p in q}):
    i = [k for k, p in enumerate(q) if p["t"] == t]
    print("  %-8s n %2d  offset x%.2f  spread x%.2f" % (t, len(i), np.exp(res[i].mean()), np.exp(res[i].std())))
print("no refit, old law F-1 = 15.5 (Lam/N)^0.59:")
old = y - np.log(15.5 * np.exp(x) ** 0.59)
for t in sorted({p["t"] for p in q}):
    i = [k for k, p in enumerate(q) if p["t"] == t]
    print("  %-8s measured/predicted x%.2f  spread x%.2f" % (t, np.exp(old[i].mean()), np.exp(old[i].std())))
yp = np.log([p["P"] - 1 for p in q])
bp, ap = np.polyfit(x, yp, 1); rp = yp - (ap + bp * x)
print("same points, P - 1 = %.2f (Lam/N)^%.2f  R2 %.3f  rms x%.2f" % (np.exp(ap), bp, 1 - rp.var() / yp.var(), np.exp(rp.std())))
big = [k for k, p in enumerate(q) if p["F"] > 2]
print("F > 2 only: n %d  rms x%.2f around pooled line" % (len(big), np.exp(res[big].std())))
Xp = np.c_[np.ones(len(q)), np.log([p["lam"] for p in q]), np.log([p["N"] for p in q])]
cp, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
print("free P law: P-1 ~ Lam^%.2f N^%.2f  R2 %.3f" % (cp[1], cp[2], 1 - (yp - Xp @ cp).var() / yp.var()))
for t in sorted({p["t"] for p in q}):
    i = [k for k, p in enumerate(q) if p["t"] == t]
    print("  P law %-8s offset x%.2f spread x%.2f" % (t, np.exp(rp[i].mean()), np.exp(rp[i].std())))
for nm, v in [("logN", np.log([p["N"] for p in q])), ("logLam", np.log([p["lam"] for p in q])), ("logr", np.log([p["r"] for p in q]))]:
    print("residual corr with %s: %.2f" % (nm, np.corrcoef(rp, v)[0, 1]),
          " within-setting:", " ".join("%s %.2f" % (t[:4], np.corrcoef(rp[[k for k, p in enumerate(q) if p["t"] == t]], v[[k for k, p in enumerate(q) if p["t"] == t]])[0, 1])
                                       for t in sorted({p["t"] for p in q})))
# local step-slope of the slower thing's own curve at the run's step count
def alpha(c, S):
    s = np.log([a for a, _ in c]); e = np.log([b for _, b in c]); i = np.searchsorted(s, np.log(S))
    lo, hi = max(0, i - 1), min(len(s) - 1, i + 1)
    return -(e[hi] - e[lo]) / (s[hi] - s[lo])
