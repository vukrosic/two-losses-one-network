"""Teacher world: P table, gap and size exponents, and the driving laws with no refit.   python3 teachfit.py [tag] [steps]"""
import json, sys, numpy as np
TAG = sys.argv[1] if len(sys.argv) > 1 else "grid"; ST = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
R = [json.loads(l) for l in open("teacher.jsonl")]
R = [r for r in R if r["tag"] == TAG and r["steps"] == ST]
cfgs = sorted({(r["hs"], r["snoise"], r["hf"], r["depth"], r["opt"], r.get("sep", False), r.get("ntrain", 0)) for r in R})
allp = []
for cf in cfgs:
    rs = [r for r in R if (r["hs"], r["snoise"], r["hf"], r["depth"], r["opt"], r.get("sep", False), r.get("ntrain", 0)) == cf]
    sp = {r["w"]: r for r in rs if r["net"] == "split"}
    gaps = sorted({r["a"] for r in rs if r["net"] == "single"})
    print("\nHS %s noise %s HF %s depth %s %s sep %s ntrain %s" % cf); print("  w     N     own   " + "  ".join("a=%-4g" % a for a in gaps))
    pts = []
    for w in sorted(sp):
        s = sp[w]; row = []
        for a in gaps:
            q = [r for r in rs if r["net"] == "single" and r["w"] == w and r["a"] == a]
            if not q: row.append("  -  "); continue
            P = q[-1]["cv"][1] / s["cv"][1]
            lam = a * a * s["cv"][0] * s["var"][0] / (s["cv"][1] * s["var"][1])
            pts.append(dict(w=w, N=s["N"], a=a, P=P, lam=lam, own=s["cv"][1], cf=cf)); row.append("%5.2f" % P)
        print("  %-4d %6d  %.4f  " % (w, s["N"], s["cv"][1]) + "  ".join(row))
    q = [p for p in pts if p["P"] > 1.05]
    if len(q) < 4: continue
    y = np.log([p["P"] - 1 for p in q])
    X = np.c_[np.ones(len(q)), np.log([p["a"] for p in q]), np.log([p["N"] for p in q])]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    ws = sorted(sp); slope = -np.polyfit(np.log([sp[w]["N"] for w in ws]), np.log([sp[w]["cv"][1] for w in ws]), 1)[0]
    r1 = y - np.log([0.149 * p["lam"] ** 0.77 for p in q]); r2 = y - np.log([20.7 * (p["lam"] / p["N"]) ** 0.52 for p in q])
    print("  P-1 ~ a^%.2f N^%.2f | own-error size slope %.2f -> rule-3 size exponent %.2f | law 0.149 Lam^0.77: x%.2f (spread x%.2f) | 20.7 (Lam/N)^0.52: x%.2f (spread x%.2f)"
          % (c[1], c[2], slope, -0.87 + 1.40 * slope, np.exp(r1.mean()), np.exp(r1.std()), np.exp(r2.mean()), np.exp(r2.std())))
    allp += q
y = np.log([p["P"] - 1 for p in allp])
X = np.c_[np.ones(len(allp)), np.log([p["lam"] for p in allp]), np.log([p["N"] for p in allp])]
c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
print("\npooled teacher: P-1 = %.3g Lam^%.2f N^%.2f  R2 %.3f  n %d" % (np.exp(c[0]), c[1], c[2], 1 - r.var() / y.var(), len(y)))
b, a0 = np.polyfit(np.log([p["lam"] for p in allp]), y, 1); r = y - a0 - b * np.log([p["lam"] for p in allp])
print("pooled teacher Lam only: P-1 = %.3g Lam^%.2f  R2 %.3f" % (np.exp(a0), b, 1 - r.var() / y.var()))
