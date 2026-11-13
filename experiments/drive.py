"""Small worlds with several moving things; one shared predictor vs one predictor per thing.

WORLD=drive (default): a car (steered by throttle and steer) and one or more pedestrians crossing the road.
  Each thing is tracked as 4 numbers (x, y, vx, vy), like an object-tracking driving stack; no pixels.
  Pedestrian rule (reacts to the car): walks from one kerb to the other at its speed, turns back at the far side,
  slows to a stop when the car is close (speed x sigmoid((distance - 0.15) / 0.03)).
  Car: speed += 0.002 * throttle (kept in 0.01..0.05), heading += 0.08 * steer (kept in -0.5..0.5); controls are a
  smooth random mix with lane keeping and braking when a pedestrian is ahead and close.
WORLD=arm: one robot with two parts: a slow base (pushed by 2 controls, with friction, bouncing off walls) and the arm
  tip (base + 0.2 x (cos angle, sin angle); a third control turns the arm). Things: base, tip.

The predictor gets the last 3 frames of every thing plus the current controls and predicts each thing's next change.

    python3 drive.py NET ARCH size [steps=6000]
NET = split | single;  ARCH = mlp | tf | gru;  size = s | m | l (single-net width 20 / 75 / 284 for mlp; the split
widths are chosen so the total weights match within a few %), or a number = that exact width (each split part gets it)
Env:
  NORM=1 scales each number's error by its typical squared change; NORM=agent one scale per thing (its mean typical
    squared change); default: one shared constant for all numbers (keeps the balance between things).
  LOSS=l1 uses absolute error instead of squared error (same scaling, square-rooted).
  PED= pedestrian speed (default 0.006); PEDS=0.003,0.012 several pedestrians with these speeds (overrides PED).
  CURVE=a  pedestrians weave: sideways velocity a x speed x cos(phase), phase += 0.3 per step (deterministic).
  PNOISE=s random push on each pedestrian's position every step (std s).
  ARMGAIN= how fast the arm turns (default 0.06).   NCLIP= training clips (default 1000).
  DECAY=1 cosine learning-rate decay to 0 (default constant 1e-3). LR= peak learning rate. BATCH= batch size (256).
  AVG=f evaluate the average of the weights over the last f of the steps.
  CARSCALE=a measure the first thing in a x bigger units (speed gap x a, same motion).
Prints held-out error per thing as error / error of a constant-velocity guess (position += velocity).
Appends to drive.jsonl in the cwd.
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
NET, ARCH, SIZE = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
E = os.environ.get
WORLD = E("WORLD", "drive")
PEDS = [float(x) for x in E("PEDS", E("PED", "0.006")).split(",")]
CAR, CURVE, PNOISE, ARMGAIN = float(E("CAR", 0.03)), float(E("CURVE", 0)), float(E("PNOISE", 0)), float(E("ARMGAIN", 0.06))
NORM, AGENTNORM, L1 = E("NORM") == "1", E("NORM") == "agent", E("LOSS") == "l1"
NCLIP, T, K = int(E("NCLIP", 1000)), 40, 3
M = 1 + len(PEDS) if WORLD == "drive" else 2           # number of things
C = 2 if WORLD == "drive" else 3                          # number of controls
NAMES = (["car"] + [f"ped{i}" if len(PEDS) > 1 else "ped" for i in range(len(PEDS))]) if WORLD == "drive" else ["base", "tip"]


def sig(x):
    return 1 / (1 + np.exp(-x))


def sim_drive(n, seed):
    rng, nrng = np.random.default_rng(seed), np.random.default_rng(seed + 100)
    lane = np.where(rng.random(n) < 0.5, 0.44, 0.56)
    car = np.stack([rng.uniform(0, 0.2, n), lane], 1)
    spd, hd = np.full(n, CAR) * rng.uniform(0.8, 1.2, n), np.zeros(n)
    P = []
    for v in PEDS:
        side = rng.random(n) < 0.5
        P.append(dict(v=v, pos=np.stack([rng.uniform(0.4, 1.0, n), np.where(side, 0.25, 0.75)], 1),
                      gy=np.where(side, 0.75, 0.25), gx=None, vel=np.zeros((n, 2)), ph=rng.uniform(0, 6.3, n)))
        P[-1]["gx"] = P[-1]["pos"][:, 0] + rng.uniform(-0.1, 0.1, n)
    cv, thr, st = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
    S, U = [], []
    for t in range(T + 1):
        S.append(np.concatenate([car, cv] + [np.concatenate([p["pos"], p["vel"]], 1) for p in P], 1))
        if t == T:
            break
        thr = 0.8 * thr + 0.4 * rng.uniform(-1, 1, n)
        st = 0.8 * st + 0.3 * rng.uniform(-1, 1, n)
        ahead = np.zeros(n, bool)
        for p in P:
            dx = p["pos"][:, 0] - car[:, 0]
            ahead |= (dx > 0) & (dx < 0.3) & (np.abs(p["pos"][:, 1] - car[:, 1]) < 0.15)
        a_thr = np.clip(np.where(ahead, -1.0, thr), -1, 1)
        a_st = np.clip(st - 3.0 * (car[:, 1] - lane) - 2.0 * hd, -1, 1)
        U.append(np.stack([a_thr, a_st], 1))
        spd = np.clip(spd + 0.002 * a_thr, 0.01, 0.05)
        hd = np.clip(hd + 0.08 * a_st, -0.5, 0.5)
        cv = np.stack([spd * np.cos(hd), spd * np.sin(hd)], 1)
        car = car + cv
        for p in P:
            d = np.linalg.norm(car - p["pos"], axis=1)
            tgt = np.stack([p["gx"], p["gy"]], 1) - p["pos"]
            dist = np.linalg.norm(tgt, axis=1)
            go = sig((d - 0.15) / 0.03)
            vel = tgt / np.maximum(dist, 1e-9)[:, None] * np.minimum(dist, p["v"] * go)[:, None]
            if CURVE:
                p["ph"] = p["ph"] + 0.3
                vel[:, 0] += CURVE * p["v"] * go * np.cos(p["ph"])
            new = p["pos"] + vel
            if PNOISE:
                new = new + nrng.normal(0, PNOISE, (n, 2))
            p["vel"], p["pos"] = new - p["pos"], new
            p["gy"] = np.where(dist < 0.01, 1 - p["gy"], p["gy"])
    return np.stack(S, 1).astype(np.float32), np.stack(U, 1).astype(np.float32)


def sim_arm(n, seed):
    rng = np.random.default_rng(seed)
    b, bv = rng.uniform(0.3, 0.7, (n, 2)), np.zeros((n, 2))
    th, om = rng.uniform(0, 6.3, n), np.zeros(n)
    u = np.zeros((n, 3))
    tip = b + 0.2 * np.stack([np.cos(th), np.sin(th)], 1); tv = np.zeros((n, 2))
    S, U = [], []
    for t in range(T + 1):
        S.append(np.concatenate([b, bv, tip, tv], 1))
        if t == T:
            break
        u = np.clip(0.8 * u + 0.4 * rng.uniform(-1, 1, (n, 3)), -1, 1)
        U.append(u.copy())
        bv = 0.9 * bv + 0.002 * u[:, :2]
        nb = b + bv
        out = (nb < 0.1) | (nb > 0.9)
        bv = np.where(out, -bv, bv); nb = np.clip(nb, 0.1, 0.9)
        bv, b = nb - b, nb
        om = 0.8 * om + ARMGAIN * u[:, 2]
        th = th + om
        nt = b + 0.2 * np.stack([np.cos(th), np.sin(th)], 1)
        tv, tip = nt - tip, nt
    return np.stack(S, 1).astype(np.float32), np.stack(U, 1).astype(np.float32)


simulate = sim_drive if WORLD == "drive" else sim_arm
D = 4 * M
IN = K * D + C


def pairs(S, U):
    S, U = torch.from_numpy(S), torch.from_numpy(U)
    js = range(K - 1, T)
    X = torch.cat([torch.cat([S[:, j - K + 1:j + 1].flatten(1), U[:, j]], 1) for j in js])
    cur = torch.cat([S[:, j] for j in js]); nxt = torch.cat([S[:, j + 1] for j in js])
    near = torch.cat([(S[:, j + 1, 0:2] - S[:, j + 1, 4:6]).norm(dim=1) < 0.25 for j in js]) if WORLD == "drive" else torch.zeros(len(cur), dtype=torch.bool)
    return X, cur, nxt, near


DEPTH = int(E("DEPTH", 2))   # hidden layers of the MLP


def mlp(i, h, o):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(DEPTH - 1):
        layers += [nn.Linear(h, h), nn.GELU()]
    m = nn.Sequential(*layers, nn.Linear(h, o))
    nn.init.normal_(m[-1].weight, std=0.01); nn.init.zeros_(m[-1].bias)
    return m


class TF(nn.Module):
    """One token per thing: its last 3 frames (12) + the controls (C) + a thing one-hot (M); 2 attention blocks with
    weights shared across tokens; the head predicts that token's 4-number change for the things in `out`."""
    def __init__(self, d, out):
        super().__init__()
        self.out = out
        self.inp = nn.Linear(12 + C + M, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, 2, 2 * d, 0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(2)])
        self.head = nn.Linear(d, 4)
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    def forward(self, X):
        n = len(X)
        hist = X[:, :K * D].view(n, K, M, 4).permute(0, 2, 1, 3).reshape(n, M, K * 4)
        tok = torch.cat([hist, X[:, None, K * D:].expand(n, M, C), torch.eye(M).expand(n, M, M)], 2)
        h = self.inp(tok)
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, self.out]).flatten(1)


class GRU(nn.Module):
    """Reads the 3 frames in order (all things' numbers each frame), then its last state + the controls go through a
    linear head to `o` outputs."""
    def __init__(self, h, o):
        super().__init__()
        self.g = nn.GRU(D, h, batch_first=True)
        self.head = nn.Linear(h + C, o)
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    def forward(self, X):
        _, h = self.g(X[:, :K * D].view(len(X), K, D))
        return self.head(torch.cat([h[0], X[:, K * D:]], 1))


def nparams(ms):
    return sum(p.numel() for m in ms for p in m.parameters())


SINGLE = {"mlp": {"xs": 8, "s": 20, "sm": 40, "m": 75, "ml": 150, "l": 284}, "tf": {"xs": 10, "s": 18, "sm": 28, "m": 40, "ml": 64, "l": 90},
          "gru": {"xs": 6, "s": 12, "sm": 22, "m": 36, "ml": 70, "l": 120}}


def make(net, w):
    if ARCH == "mlp":
        return [mlp(IN, w, 4) for _ in range(M)] if net == "split" else [mlp(IN, w, D)]
    if ARCH == "tf":
        return [TF(w, [i]) for i in range(M)] if net == "split" else [TF(w, list(range(M)))]
    return [GRU(w, 4) for _ in range(M)] if net == "split" else [GRU(w, D)]


def build(net):
    if SIZE.isdigit():   # size = an explicit width (every part of a split net gets this width)
        torch.manual_seed(0)
        return (lambda ns: (ns, (lambda X: torch.cat([m(X) for m in ns], 1)) if net == "split" else ns[0], int(SIZE)))(make(net, int(SIZE)))
    h = SINGLE[ARCH][SIZE]
    torch.manual_seed(0)
    target = nparams(make("single", h))
    if net == "single":
        w = h
    else:   # split width whose total weights is closest to the single net's
        step = 2 if ARCH == "tf" else 1
        w = min(range(step * 2, 2 * h + 2, step), key=lambda w: abs(nparams(make("split", w)) - target))
    torch.manual_seed(0)
    ns = make(net, w)
    f = (lambda X: torch.cat([m(X) for m in ns], 1)) if net == "split" else ns[0]
    return ns, f, w


CARSCALE = float(E("CARSCALE", 1))   # measure thing SCALETHING (default 0 = car) in units CARSCALE x bigger: the speed
SCALETHING = int(E("SCALETHING", 0)) # gap grows by CARSCALE while the motion and the other things stay identical
TAG = E("TAG", "")


def rescale(p):
    X, cur, nxt, near = p
    if CARSCALE != 1:
        X, cur, nxt = X.clone(), cur.clone(), nxt.clone()
        o = 4 * SCALETHING
        for fr in range(K):
            X[:, fr * D + o:fr * D + o + 4] *= CARSCALE
        cur[:, o:o + 4] *= CARSCALE; nxt[:, o:o + 4] *= CARSCALE
    return X, cur, nxt, near


t0 = time.time()
tr = rescale(pairs(*simulate(NCLIP, 1)))
te = rescale(pairs(*simulate(200, 2)))
ch = (tr[2] - tr[1]).pow(2).mean(0)
per = [ch[4 * i:4 * i + 4].mean() for i in range(M)]
scale = ch if NORM else torch.cat([p.expand(4) for p in per]) if AGENTNORM else ch.mean().expand(D)
moves = [float(p.sqrt()) for p in per]
ns, f, width = build(NET)
LW = [float(x) for x in E("LW", ",".join(["1"] * M)).split(",")]   # loss weight per thing
OPT = E("OPT", "adam")   # adam | sgd (momentum 0.9, lr SGDLR) | muon (hidden h x h matrices, lr MUONLR; the rest Adam)


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    Xn = G / (G.norm() + 1e-7)
    tr_ = Xn.shape[0] > Xn.shape[1]
    if tr_: Xn = Xn.T
    for _ in range(steps):
        A = Xn @ Xn.T
        Xn = a * Xn + (b * A + c * A @ A) @ Xn
    return Xn.T if tr_ else Xn


class Muon(torch.optim.Optimizer):
    def __init__(self, ps, lr=0.02, momentum=0.95):
        super().__init__(ps, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            for p in gr["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "buf" not in st: st["buf"] = torch.zeros_like(p)
                buf = st["buf"]; buf.mul_(gr["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad + gr["momentum"] * buf)
                p.add_(u, alpha=-gr["lr"] * max(1, p.shape[0] / p.shape[1]) ** 0.5)


class Multi:
    def __init__(self, opts): self.opts = opts; self.param_groups = [g for o in opts for g in o.param_groups]
    def zero_grad(self): [o.zero_grad() for o in self.opts]
    def step(self): [o.step() for o in self.opts]


allp = [p for m in ns for p in m.parameters()]
if OPT == "sgd":
    opt = torch.optim.SGD(allp, lr=float(E("SGDLR", 0.03)), momentum=0.9)
elif OPT == "muon":
    hid = [p for m in ns for mod in m.modules() if isinstance(mod, nn.Linear) and mod.in_features == mod.out_features for p in [mod.weight]]
    ids = {id(p) for p in hid}
    opt = Multi([Muon(hid, lr=float(E("MUONLR", 0.02))), torch.optim.Adam([p for p in allp if id(p) not in ids], lr=float(E("LR", 1e-3)))])
else:
    opt = torch.optim.Adam(allp, lr=float(E("LR", 1e-3)))
for pg in opt.param_groups:
    pg["base"] = pg["lr"]
X, cur, nxt, _ = tr
mu, sd = X.mean(0), X.std(0) + 1e-6       # standardise inputs only
g = torch.Generator().manual_seed(0)
LR = float(E("LR", 1e-3))
DECAY = E("DECAY") == "1"   # cosine learning-rate decay from LR to 0 over the run (default: constant 1e-3)
BATCH = int(E("BATCH", 256))
AVG = float(E("AVG", 0))    # average the weights over the last AVG fraction of steps and evaluate the average
params = [p for m in ns for p in m.parameters()]
avg, navg = [torch.zeros_like(p) for p in params], 0
for s in range(STEPS):
    if DECAY:
        for pg in opt.param_groups:
            pg["lr"] = pg["base"] * 0.5 * (1 + np.cos(np.pi * s / STEPS))
    j = torch.randint(0, len(X), (BATCH,), generator=g)
    r = cur[j] + f((X[j] - mu) / sd) - nxt[j]
    e = r.abs() / scale.sqrt() if L1 else r.pow(2) / scale
    loss = sum(LW[i] * e[:, 4 * i:4 * i + 4].mean() for i in range(M)) / M
    opt.zero_grad(); loss.backward(); opt.step()
    if AVG and s >= STEPS * (1 - AVG):
        with torch.no_grad():
            for a, p in zip(avg, params):
                a += p
        navg += 1
if AVG:
    with torch.no_grad():
        for a, p in zip(avg, params):
            p.copy_(a / navg)
with torch.no_grad():
    Xe, ce, ne, near = te
    err = (ce + f((Xe - mu) / sd) - ne).pow(2)
    cvg = ce.clone()
    for i in range(M):
        cvg[:, 4 * i:4 * i + 2] += ce[:, 4 * i + 2:4 * i + 4]
    base_cv = (cvg - ne).pow(2)
    cvm = [float(err[:, 4 * i:4 * i + 4].sum() / base_cv[:, 4 * i:4 * i + 4].sum()) for i in range(M)]
    mse = [float(err[:, 4 * i:4 * i + 4].mean() / ch.mean()) for i in range(M)]       # held-out error in loss units (shared constant)
    cvb = [float(base_cv[:, 4 * i:4 * i + 4].mean() / ch.mean()) for i in range(M)]   # constant-velocity guess error, same units
    cv_near = float(err[near][:, 4:8].sum() / base_cv[near][:, 4:8].sum()) if WORLD == "drive" else None
res = dict(world=WORLD, net=NET, arch=ARCH, size=SIZE, steps=STEPS, decay=DECAY, lr=LR, batch=BATCH, avg=AVG, norm="number" if NORM else "agent" if AGENTNORM else "none",
           loss="l1" if L1 else "l2", peds=PEDS, ped=PEDS[0], car=CAR, curve=CURVE, pnoise=PNOISE, armgain=ARMGAIN, nclip=NCLIP, carscale=CARSCALE, scalething=SCALETHING, tag=TAG, depth=DEPTH, opt=OPT, lw=LW,
           sgdlr=float(E("SGDLR", 0.03)) if OPT == "sgd" else None, muonlr=float(E("MUONLR", 0.02)) if OPT == "muon" else None,
           width=width, params=nparams(ns), names=NAMES, moves=moves, cv=cvm, mse=mse, cv_base=cvb, cv_near=cv_near, seconds=round(time.time() - t0, 1))
if WORLD == "drive" and M == 2:   # field names used by the first driving runs
    res.update(c_car=moves[0], c_ped=moves[1], cv_car=cvm[0], cv_ped=cvm[1], cv_ped_near=cv_near)
tag = "agent " if AGENTNORM else "scaled" if NORM else "plain "
print(f"{WORLD} {ARCH} {NET:6s} {SIZE} {tag}{' l1' if L1 else ''} w {width} params {res['params']:6d} | moves "
      + " ".join(f"{n} {m:.4f}" for n, m in zip(NAMES, moves)) + " | err/const-vel " + " ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, cvm))
      + (f" near {cv_near:.3f}" if cv_near is not None else "") + f" | {res['seconds']} s")
with open("drive.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
