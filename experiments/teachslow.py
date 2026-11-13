"""Teacher world, fixed-slowdown test: F = 3000 / (steps the own student needs to reach the shared student's slow error).
    python3 teachslow.py"""
import json, numpy as np
R = [json.loads(l) for l in open("teacher.jsonl")]
R = [r for r in R if r["tag"] == "grid"]
cfg = lambda r: (r["hs"], r["snoise"], r.get("sep", False), r.get("ntrain", 0))
pts = []
for c in sorted({cfg(r) for r in R}):
    rs = [r for r in R if cfg(r) == c]
    for w in sorted({r["w"] for r in rs}):
        cur = sorted({r["steps"]: r["cv"][1] for r in rs if r["net"] == "split" and r["w"] == w}.items())
        if len(cur) < 4: continue
        s, e = np.log([a for a, _ in cur]), np.log([b for _, b in cur])
        sp = next(r for r in rs if r["net"] == "split" and r["w"] == w and r["steps"] == 3000)
        i = list(np.exp(s).round()).index(3000); alpha = -(e[i + 1] - e[i - 1]) / (s[i + 1] - s[i - 1])
        for r in rs:
            if r["net"] != "single" or r["w"] != w or r["steps"] != 3000: continue
            E = r["cv"][1]; P = E / sp["cv"][1]
            lam = r["a"] ** 2 * sp["cv"][0] * sp["var"][0] / (sp["cv"][1] * sp["var"][1])
            mono = np.all(np.diff(e) < 0)
            if mono and e.min() <= np.log(E) <= e.max():
                F = 3000 / np.exp(np.interp(-np.log(E), -e, s))
            elif np.log(E) > e.max():
                F = 3000 / 750 * np.exp((np.log(E) - e[0]) / max(-(e[1] - e[0]) / (s[1] - s[0]), 1e-3))   # extrapolate below 750 steps
            else:
                F = np.nan
            pts.append(dict(c=c, w=w, N=sp["N"], a=r["a"], P=P, lam=lam, F=F, alpha=alpha))
for c in sorted({p["c"] for p in pts}):
    q = [p for p in pts if p["c"] == c]
    print("HS %s noise %s sep %s ntrain %s: alpha %s" % (*c, " ".join("%.2f" % a for a in sorted({round(p['alpha'], 2) for p in q}))))
q = [p for p in pts if p["P"] > 1.05 and np.isfinite(p["F"]) and p["F"] > 1.05]
lam = np.log([p["lam"] for p in q])
for nm, y in [("P-1", np.log([p["P"] - 1 for p in q])), ("F-1", np.log([p["F"] - 1 for p in q]))]:
    b, a = np.polyfit(lam, y, 1); res = y - a - b * lam
    print("%s = %.3g Lam^%.2f  R2 %.3f  rms x%.2f  n %d" % (nm, np.exp(a), b, 1 - res.var() / y.var(), np.exp(res.std()), len(q)))
    for c in sorted({p["c"] for p in q}):
        i = [k for k, p in enumerate(q) if p["c"] == c]
        print("    HS %-3s noise %-3s sep %-5s ntrain %-4s offset x%.2f" % (*c, np.exp(res[i].mean())))
pp = [p for p in pts if np.isfinite(p["F"]) and p["F"] > 1]
pred = np.array([p["F"] ** p["alpha"] for p in pp]); meas = np.array([p["P"] for p in pp])
print("P vs F^alpha: median measured/predicted %.2f, middle half %.2f-%.2f" % (np.median(meas / pred), *np.percentile(meas / pred, [25, 75])))
