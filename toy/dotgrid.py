"""Dot-world gap x size grid (LATSCALE runs).  P = stranger miss shared / own; Lam = robot loss / stranger loss from the split nets."""
import json, numpy as np
R = [json.loads(l) for l in open("two_vs_one.jsonl")]
R = [r for r in R if "latscale" in r and r.get("decay") and r["steps"] == 6000 and r.get("latthing", 0) == 0 and r.get("opt", "adam") == "adam"]
sp = {r["widths"][0]: r for r in R if r["net"] == "split" and r["latscale"] == 1}
pts = []
print("w(split)  N      own  " + "  ".join("a=%-5s" % a for a in [0.5, 1, 2, 4, 8]))
for w in sorted(sp):
    s = sp[w]; own = s["test"][1]; row = []
    for a in [0.5, 1, 2, 4, 8]:
        q = [r for r in R if r["net"] == "single" and r["widths"][0] == w and r["latscale"] == a]
        if not q: row.append("  -  "); continue
        r = q[-1]; P = r["test"][1] / own
        ch = [r["ch_robot"], r["ch_stranger"]]
        lam = (s["test"][0] * ch[0] ** 2 * a ** 2 / s["ch_robot"] ** 2 * s["ch_robot"] ** 2) / (own * ch[1] ** 2)   # split robot miss is in unscaled units: same relative miss
        lam = (s["test"][0] * ch[0] ** 2) / (own * ch[1] ** 2)
        pts.append(dict(w=w, N=r["params"], a=a, r=ch[0] / ch[1], P=P, lam=lam)); row.append("%5.2f" % P)
    print("%-8d %6d  %.3f  " % (w, s["params"], own) + "  ".join(row))
q = [p for p in pts if p["P"] > 1.05]
y = np.log([p["P"] - 1 for p in q])
for names in [["lam"], ["lam", "N"], ["r", "N"]]:
    X = np.c_[np.ones(len(q)), np.log([[p[n] for n in names] for p in q])]
    c, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c
    print("dot world: P-1 = %.3g " % np.exp(c[0]) + " ".join("%s^%.2f" % (n, v) for n, v in zip(names, c[1:])) + "  R2 %.2f" % (1 - res.var() / y.var()))
res = y - np.log([0.149 * p["lam"] ** 0.77 for p in q]); res2 = y - np.log([7.27 * p["lam"] ** 0.68 * p["N"] ** -0.45 for p in q])
print("driving law 0.149 Lam^0.77: measured/predicted median x%.2f, spread x%.2f;  with N: x%.2f spread x%.2f" % (np.exp(np.median(res)), np.exp(res.std()), np.exp(np.median(res2)), np.exp(res2.std())))
res3 = y - np.log([20.7 * (p["lam"] / p["N"]) ** 0.52 for p in q])
print("driving Lam/N law 20.7 (Lam/N)^0.52: measured/predicted median x%.2f, spread x%.2f, n %d" % (np.exp(np.median(res3)), np.exp(res3.std()), len(q)))
for w in sorted({p["w"] for p in q}):
    i = [k for k, p in enumerate(q) if p["w"] == w]
    print("   w%d  x%.2f" % (w, np.exp(np.mean(res3[i]))))
