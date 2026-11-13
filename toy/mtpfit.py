"""Scores the multi-token prediction runs (mtp.jsonl) against the predictions written in LAB-NOTES.md (finding 23).

    python3 mtpfit.py

Each head i is one "thing" with loss term w_i * L_i (w = 1 for the next-token head, W / (K - 1) for each future head).
Lambda_i = (sum of the other terms) / (head i's term), each L from that head's own model (net=own) at the same width.
P_i = head i's excess loss in the shared model / its excess loss in its own model.
Law, no refit: P - 1 = 0.15 * Lambda^0.75 when Lambda > 1, P = 1 otherwise.
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(H, "mtp.jsonl"))]
cfg = lambda r, steps=None: (r["V"], r["order"], r["alpha"], r["zipf"], r["layers"], r["headblock"], steps or r["steps"])
own = {(cfg(r), r["K"], r["d"]): r for r in R if r["net"] == "own"}
shared = [r for r in R if r["net"] == "shared"]
law = lambda L: 1 + 0.15 * L ** 0.75 if L > 1 else 1.0


def lam(r, field, steps=None, frac="1"):
    """Lambda for each head of shared run r, from the own models; None if an own model is missing"""
    K, W, d = r["K"], r["W"], r["d"]
    w = {k: 1.0 if k == 1 else W / (K - 1) for k in r["offs"]}
    L = {}
    for k in r["offs"]:
        o = own.get((cfg(r, steps), k, d))
        if o is None: return None
        L[k] = o["curve"][frac][str(k)][field]
    term = {k: w[k] * L[k] for k in L}
    return {k: (sum(term.values()) - term[k]) / term[k] for k in term}


def P(r, k, frac="1"):
    o = own.get((cfg(r), k, r["d"]))
    return r["curve"][frac][str(k)]["excess"] / o["curve"][frac][str(k)]["excess"] if o else None


def fit(pts, name):
    """pts: (Lambda, P); fit log(P - 1) = c + x log Lambda over points with Lambda > 1 and P > 1"""
    p = [(L, q) for L, q in pts if L and q and L > 1 and q > 1]
    if len(p) < 3:
        print(f"  {name}: {len(p)} usable points, no fit"); return
    x, y = np.log([L for L, _ in p]), np.log([q - 1 for _, q in p])
    b, a = np.polyfit(x, y, 1); r2 = 1 - ((y - a - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    off = np.exp(np.median(y - np.log([law(L) - 1 for L, _ in p])))
    inside = sum(abs(yy - np.log(law(L) - 1)) <= np.log(2.5) for (L, _), yy in zip(p, y))
    print(f"  {name}: P-1 = {np.exp(a):.3f} * Lambda^{b:.2f} (R2 {r2:.2f}); fixed law offset x{off:.2f}; "
          f"within x2.5 of the fixed law in {inside}/{len(p)}")


print("Prediction 1-2: next-token head (P1) and future heads, Lambda from excess loss, full training")
print(f"{'K':>2} {'d':>4} {'W':>5} {'N':>7} {'Lam_ex':>8} {'Lam_raw':>8} {'P1':>6} {'law':>6} {'P1_eqN':>7}  future heads P (Lambda)")
pts_ex, pts_raw, pts_fut = [], [], []
for r in sorted(shared, key=lambda r: (cfg(r), r["K"], r["d"], r["W"])):
    le, lr = lam(r, "excess"), lam(r, "ce")
    if le is None: continue
    p1 = P(r, 1)
    # equal-weight comparison: own next-token models fitted over width, read at the shared model's weight count
    o1 = [o for (c, k, d), o in own.items() if c == cfg(r) and k == 1]
    peq = None
    if len(o1) >= 2:
        b, a = np.polyfit(np.log([o["N"] for o in o1]), np.log([o["final"]["1"]["excess"] for o in o1]), 1)
        peq = r["final"]["1"]["excess"] / np.exp(a + b * np.log(r["N"]))
    fut = [(P(r, k), le[k]) for k in r["offs"][1:]]
    pts_ex.append((le[1], p1)); pts_raw.append((lr[1], p1)); pts_fut += [(L, q) for q, L in fut]
    print(f"{r['K']:>2} {r['d']:>4} {r['W']:>5} {r['N']:>7} {le[1]:>8.2f} {lr[1]:>8.2f} {p1:>6.2f} {law(le[1]):>6.2f} "
          f"{peq if peq else float('nan'):>7.2f}  " + " ".join(f"{q:.2f}({L:.2f})" for q, L in fut)
          + ("  headblock" if r["headblock"] else "") + (f"  alpha {r['alpha']}" if r["alpha"] != 0.5 else "")
          + (f"  steps {r['steps']}" if r["steps"] != 1500 else ""))

print("\nPrediction 1 and 3: which Lambda fits")
fit(pts_ex, "next-token head, Lambda from excess loss")
fit(pts_raw, "next-token head, Lambda from raw cross-entropy")
fit(pts_fut, "future heads, Lambda from excess loss")

print("\nPrediction 2: next-token head when Lambda < 1 (law: P = 1)")
low = [q for L, q in pts_ex if L <= 1]
print(f"  {len(low)} runs, P1 = " + ", ".join(f"{q:.2f}" for q in low) if low else "  none")

print("\nPrediction 4: cheap prediction from 1/8 of training")
for r in shared:
    q = r["steps"] // 8
    short = next((s for s in shared if s["steps"] == q and (s["K"], s["d"], s["W"]) == (r["K"], r["d"], r["W"])
                  and cfg(s, r["steps"]) == cfg(r)), None)
    l_short, l_full = lam(r, "excess", steps=q), lam(r, "excess")
    if short is None or l_short is None or l_full is None: continue
    p_short, p_full = P(short, 1), P(r, 1)
    print(f"  K {r['K']} d {r['d']} W {r['W']}: Lambda 1/8-run {l_short[1]:.2f} vs full {l_full[1]:.2f} (x{l_short[1] / l_full[1]:.2f}); "
          f"P1 1/8-run {p_short:.2f} vs full {p_full:.2f}; law from 1/8-run Lambda {law(l_short[1]):.2f}")

print("\nPrediction 5: does P1 fall with width? (own next-token excess slope over width alongside)")
grp = defaultdict(list)
for r in shared: grp[(cfg(r), r["K"], r["W"])].append(r)
for (c, K, Wt), rs in sorted(grp.items()):
    rs = [r for r in sorted(rs, key=lambda r: r["d"]) if P(r, 1)]
    o1 = sorted([o for (cc, k, d), o in own.items() if cc == c and k == 1], key=lambda o: o["N"])
    if len(rs) < 2 or len(o1) < 2: continue
    so = np.polyfit(np.log([o["N"] for o in o1]), np.log([o["final"]["1"]["excess"] for o in o1]), 1)[0]
    print(f"  K {K} W {Wt} steps {c[-1]}: P1 over d {[r['d'] for r in rs]} = {[round(P(r, 1), 2) for r in rs]}; "
          f"own next-token excess ~ N^{so:.2f}")
