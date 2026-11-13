"""Rules table for finding 21: per setting, gap exponent and size exponent of P - 1, the slower thing's own-error size slope,
and the loss-ratio law's offset (measured / 0.149 Lam^0.77).   python3 rules.py"""
import json, numpy as np, allfit
DRIVE = [("base:1", "driving MLP, Adam, 6k decayed"), ("fine:1", "driving, fine gap grid"), ("long18k:1", "driving, 18k steps"),
         ("nodecay:1", "driving, constant LR"), ("lr3e-3:1", "driving, LR 3e-3"), ("lr3e-4:1", "driving, LR 3e-4"),
         ("batch64:1", "driving, batch 64"), ("clip250:1", "driving, 250 clips"), ("ped12:1", "driving, faster pedestrian"),
         ("weave:1", "driving, weaving pedestrian"), ("twoped:1", "two pedestrians (1st)"), ("twoped:2", "two pedestrians (2nd)"),
         ("pedfast:0", "driving, car is the slower thing"), ("arm:0", "arm robot"), ("armfast:0", "arm robot, faster"),
         ("l1:1", "driving, L1 loss"), ("muon:1", "driving, Muon optimizer"), ("depth3:1", "MLP depth 3"), ("depth4:1", "MLP depth 4"),
         ("grubig:1", "GRU (sizes it can learn)"), ("agentnorm:1", "per-thing loss scaling")]
def fitrow(q):
    y = np.log([p["P1"] for p in q]); X = np.c_[np.ones(len(q)), np.log([p["r"] for p in q]), np.log([p["N"] for p in q])]
    c, *_ = np.linalg.lstsq(X, y, rcond=None); return c[1], c[2]
print("| Setting | gap exponent (P-1 ~ r^x) | size exponent (P-1 ~ N^y) | slower thing's own error vs size | law offset |")
print("|---|---|---|---|---|")
for t, name in DRIVE:
    allfit.KEEPL = t.startswith("grubig")
    pts = allfit.points(t); q = [p for p in pts if p["P1"] > 0.1]
    own = {p["N"]: p["es"] for p in pts}; Ns = sorted(own)
    aN = -np.polyfit(np.log(Ns), np.log([own[n] for n in Ns]), 1)[0]
    if len(q) < 6:
        print("| %s | flat (P-1 > 0.1 in %d of %d runs) | - | %.2f | - |" % (name, len(q), len(pts), aN)); continue
    gx, sy = fitrow(q)
    off = np.exp(np.median(np.log([p["P1"] for p in q]) - np.log([0.149 * p["lam"] ** 0.77 for p in q])))
    print("| %s | %.2f | %.2f | %.2f | x%.2f |" % (name, gx, sy, aN, off))
# dot world
R = [json.loads(l) for l in open("two_vs_one.jsonl")]
for tag, ntraj, name in [("enc_chase", 1000, "dot world, Chase stranger"), ("enc_chase", 250, "dot world, Chase, 250 clips"), ("enc_switch", 1000, "dot world, Switch stranger")]:
    rs = [r for r in R if r.get("latscale") and r["tag"] == tag and r["ntraj"] == ntraj and r["steps"] == 6000 and r.get("decay")]
    sp = {r["widths"][0]: r for r in rs if r["net"] == "split"}
    q, own = [], {}
    for r in rs:
        if r["net"] != "single" or r["widths"][0] not in sp: continue
        s = sp[r["widths"][0]]; o = s["test"][1]; own[s["params"]] = o
        P1 = r["test"][1] / o - 1
        lam = s["test"][0] * r["ch_robot"] ** 2 / (o * r["ch_stranger"] ** 2)
        if P1 > 0.1: q.append(dict(P1=P1, r=r["ch_robot"] / r["ch_stranger"], N=r["params"], lam=lam))
    Ns = sorted(own); aN = -np.polyfit(np.log(Ns), np.log([own[n] for n in Ns]), 1)[0]
    gx, sy = fitrow(q)
    off = np.exp(np.median(np.log([p["P1"] for p in q]) - np.log([0.149 * p["lam"] ** 0.77 for p in q])))
    print("| %s | %.2f | %.2f | %.2f | x%.2f |" % (name, gx, sy, aN, off))
