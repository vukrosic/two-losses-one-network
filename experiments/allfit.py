"""Pooled regressions of log(P-1) on candidate variables over many scenarios.
    python3 allfit.py [TAG:slow ...]
Variables per shared run: Lam (loss ratio from own networks), N (weights), r (move ratio),
es = slower thing's own error relative to its constant-velocity guess, ef = same for the faster thing."""
import json, sys, numpy as np
KEEPL = False
R = [json.loads(l) for l in open("drive.jsonl")]; R = [r for r in R if "carscale" in r]
SC = ["base:1","fine:1","long18k:1","arm:0","armfast:0","weave:1","twoped:1","twoped:2","l1:1","lr3e-3:1","lr3e-4:1",
      "batch64:1","clip250:1","ped12:1"]
def points(t):
    tag, slow = t.split(":"); slow = int(slow)
    rows = [r for r in R if (r.get("tag") or "base") == tag]
    split = {}
    for r in rows:
        if r["net"] == "split": split[r["size"]] = r
    out = []
    for r in rows:
        if r["net"] != "single" or r["size"] not in split or (r["size"] == "l" and not KEEPL): continue
        f = r.get("scalething", 0); sp = split[r["size"]]
        if f == slow: f = 1 - slow
        es, ef = sp["cv"][slow], sp["cv"][f]
        Es, Ef = es * r["cv_base"][slow], ef * r["cv_base"][f]
        lam = np.sqrt(Ef / Es) if r["loss"] == "l1" else (Ef / r["moves"][f]**2) / (Es / r["moves"][slow]**2) if r["norm"] == "agent" else Ef / Es
        out.append(dict(t=t, lam=lam, N=r["params"], r=r["moves"][f] / r["moves"][slow], es=es, ef=ef,
                        P1=r["cv"][slow] / es - 1, size=r["size"], steps=r["steps"], slow=slow))
    return out
if __name__ == "__main__":
    tags = sys.argv[1:] or SC
    pts = [p for t in tags for p in points(t) if p["P1"] > 0.05]
    y = np.log([p["P1"] for p in pts])
    def reg(names):
        X = np.c_[np.ones(len(y))] if not names else np.c_[np.ones(len(y)), np.log([[p[n] for n in names] for p in pts])]
        c, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c
        r2 = 1 - (res**2).sum() / ((y - y.mean())**2).sum()
        worst = max(abs(np.mean([res[i] for i, p in enumerate(pts) if p["t"] == t])) for t in tags)
        print("%-22s R2 %.3f  worst scenario x%.2f  " % ("+".join(names), r2, np.exp(worst)) +
              "  ".join("%s^%.2f" % (n, v) for n, v in zip(names, c[1:])) + "  c=%.3g" % np.exp(c[0]))
    for names in [["lam"], ["lam", "N"], ["r", "N"], ["lam", "es"], ["lam", "ef"], ["lam", "es", "ef"], ["lam", "N", "es"],
                  ["r", "es", "ef"], ["r", "ef"], ["lam", "N", "es", "ef"]]:
        reg(names)
