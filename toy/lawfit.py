"""Loss-ratio law: P - 1 = c * Lambda^b, pooled over scenarios.

    python3 lawfit.py [--fit TAG:slow,...] [--test TAG:slow,...] [--min 0.05]

Lambda = the faster thing's loss term / the slower thing's loss term, each evaluated with the error of its OWN network
(the split run at the same total size), under the loss actually used:
  squared error, shared constant scale:  Lambda = E_fast / E_slow                  (E = mean squared error)
  absolute error (LOSS=l1):              Lambda = sqrt(E_fast / E_slow)           (RMS ratio stands in for mean |e|)
  per-thing scale (NORM=agent):          Lambda = (E_fast / m_fast^2) / (E_slow / m_slow^2)   (m = RMS move)
P = slower thing's error in the shared network / in its own network.
--fit scenarios set c and b; --test scenarios are scored with that line only (prediction, no refit).
"""
import json, sys
import numpy as np

args = sys.argv[1:]
opt = {k: v for k, v in zip(args[::2], args[1::2])}
FIT = opt.get("--fit", "base:1,long18k:1,arm:0,weave:1,twoped:1,twoped:2,l1:1,fine:1").split(",")
TEST = [t for t in opt.get("--test", "").split(",") if t]
MIN = float(opt.get("--min", 0.05))
R = [json.loads(l) for l in open("drive.jsonl")]
R = [r for r in R if "carscale" in r]


def points(t):
    tag, slow = t.split(":"); slow = int(slow)
    rows = [r for r in R if (r.get("tag") or "base") == tag]
    split = {r["size"]: r for r in rows if r["net"] == "split"}
    out = []
    for r in rows:
        if r["net"] != "single" or r["size"] not in split or r["size"] == "l":
            continue
        f = r.get("scalething", 0); sp = split[r["size"]]
        own = sp["cv"][slow]
        es, ef = own * r["cv_base"][slow], sp["cv"][f] * r["cv_base"][f]
        if r["loss"] == "l1":
            lam = np.sqrt(ef / es)
        elif r["norm"] == "agent":
            lam = (ef / r["moves"][f] ** 2) / (es / r["moves"][slow] ** 2)
        else:
            lam = ef / es
        out.append(dict(t=t, lam=lam, P1=r["cv"][slow] / own - 1, N=r["params"], r=r["moves"][f] / r["moves"][slow],
                        size=r["size"]))
    return out


fp = [p for t in FIT for p in points(t) if p["P1"] > MIN]
x, y = np.log([p["lam"] for p in fp]), np.log([p["P1"] for p in fp])
b, a = np.polyfit(x, y, 1)
r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"fit on {len(fp)} points: P - 1 = {np.exp(a):.3f} * Lambda^{b:.2f}   R2 {r2:.3f}   "
      f"P = 1.5 at Lambda = {(0.5 / np.exp(a)) ** (1 / b):.1f}")
for t in FIT + TEST:
    q = points(t)
    qq = [p for p in q if p["P1"] > MIN]
    if not qq:
        print(f"  {t:14s} no points with P-1 > {MIN}; Lambda {min(p['lam'] for p in q):.2f}-{max(p['lam'] for p in q):.2f}, "
              f"predicted P-1 up to {np.exp(a) * max(p['lam'] for p in q) ** b:.2f}, measured max {max(p['P1'] for p in q):.2f}")
        continue
    xx, yy = np.log([p["lam"] for p in qq]), np.log([p["P1"] for p in qq])
    res = yy - (a + b * xx)
    print(f"  {t:14s} {'TEST' if t in TEST else 'fit '} n {len(qq):2d}/{len(q):2d}  measured / predicted x{np.exp(res.mean()):.2f}"
          f"  spread x{np.exp(res.std()):.2f}" + (f"  own slope {np.polyfit(xx, yy, 1)[0]:.2f}" if len(qq) > 3 else ""))
