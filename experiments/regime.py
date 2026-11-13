"""Size effect at fixed loss ratio vs how fast the own errors still fall with size, every setting.
    python3 regime.py
c = exponent of N in log(P-1) = a + b log Lam + c log N, fitted per setting (P-1 > 0.05).
sigma = -(slope of log own error vs log N) for the slower thing (own networks), over the sizes in that setting."""
import json, sys, numpy as np
import allfit
rows = []
def add(name, pts, own):
    q = [p for p in pts if p["P1"] > 0.05]
    if len(q) < 6 or len({p["N"] for p in q}) < 3: return
    y = np.log([p["P1"] for p in q]); X = np.c_[np.ones(len(q)), np.log([p["lam"] for p in q]), np.log([p["N"] for p in q])]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    Ns = sorted(own); sig = -np.polyfit(np.log(Ns), np.log([own[n] for n in Ns]), 1)[0]
    rows.append((name, sig, c[1], c[2], len(q)))
for t in allfit.SC + ["pedfast:0", "muon:1", "nodecay:1"]:
    pts = allfit.points(t)
    if not pts: continue
    own = {p["N"]: p["es"] for p in pts}
    add("drive " + t, pts, own)
T = [json.loads(l) for l in open("teacher.jsonl")]
cfg = lambda r: (r["tag"], r["steps"], r["hs"], r.get("snoise", 0), r.get("sep", False), r.get("ntrain", 0))
for c in sorted({cfg(r) for r in T if r["tag"] in ("grid", "short") and r["net"] == "single"}):
    rs = [r for r in T if cfg(r) == c]; sp = {r["w"]: r for r in rs if r["net"] == "split"}
    pts = [dict(P1=r["cv"][1] / sp[r["w"]]["cv"][1] - 1, N=sp[r["w"]]["N"],
                lam=r["a"] ** 2 * sp[r["w"]]["cv"][0] * sp[r["w"]]["var"][0] / (sp[r["w"]]["cv"][1] * sp[r["w"]]["var"][1]))
           for r in rs if r["net"] == "single" and r["w"] in sp]
    add("teacher steps %d HS %d noise %g sep %d ntrain %d" % c[1:], pts, {s["N"]: s["cv"][1] for s in sp.values()})
print("%-48s %6s %6s %6s %3s" % ("setting", "sigma", "b(Lam)", "c(N)", "n"))
for r in sorted(rows, key=lambda r: r[1]): print("%-48s %6.2f %6.2f %6.2f %3d" % r)
s, c = np.array([r[1] for r in rows]), np.array([r[3] for r in rows])
k, a = np.polyfit(s, c, 1)
print("c = %.2f + %.2f sigma   corr %.2f   n %d" % (a, k, np.corrcoef(s, c)[0, 1], len(rows)))
for lo, hi in [(-1, 0.1), (0.1, 0.25), (0.25, 2)]:
    z = c[(s >= lo) & (s < hi)]; print("sigma in [%.2f, %.2f): n %d, c mean %.2f, range %.2f to %.2f" % (lo, hi, len(z), z.mean(), z.min(), z.max()))
