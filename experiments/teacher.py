"""Teacher-student world: no physics, two targets made by fixed random networks.

    python3 teacher.py split|single w [steps]
env: A (unit scale of the fast target, default 1), HF / HS (hidden width of the fast / slow teacher, default 32 / 32),
     DEPTH (student hidden layers, 2), LR (1e-3), BATCH (256), OPT (adam|muon), MUONLR, SLOWAMP (slow target std, 1), TAG
Target f = A * teacher_f(x) and slow target s = SLOWAMP * teacher_s(x), each 4 numbers, x ~ N(0, I_16), fresh samples
every step (no overfitting). split = two students of width w, one per target; single = one student, width chosen so the
weight count matches. Score: held-out mean squared error of each target / its variance (cv). Appends to teacher.jsonl.
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn

E = os.environ.get
torch.set_num_threads(2)
NET, W = sys.argv[1], int(sys.argv[2]); STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
A, HF, HS, DEPTH = float(E("A", 1)), int(E("HF", 32)), int(E("HS", 32)), int(E("DEPTH", 2))
LR, BATCH, OPT, SLOWAMP = float(E("LR", 1e-3)), int(E("BATCH", 256)), E("OPT", "adam"), float(E("SLOWAMP", 1))
NTRAIN = int(E("NTRAIN", 0))   # >0: fixed training set of this many samples (finite data, as in driving); 0: fresh samples
SNOISE = float(E("SNOISE", 0))   # noise added to the slow target (train and test): a floor no capacity removes
IN, O = 16, 4
g = torch.Generator().manual_seed(0)


SEP = E("SEP") == "1"   # fast teacher reads x[:8] only, slow teacher x[8:] only (separate inputs, as in driving)


def teacher(h, cols, depth=2):
    k = len(cols)
    Ws = [torch.randn(k, h, generator=g) / k ** 0.5] + [torch.randn(h, h, generator=g) / h ** 0.5 for _ in range(depth - 1)]
    Wo = torch.randn(h, O, generator=g) / h ** 0.5
    def f(x):
        x = x[:, cols]
        for Wm in Ws:
            x = torch.tanh(2 * x @ Wm)
        return x @ Wo
    return f


tf, ts = (teacher(HF, list(range(8))), teacher(HS, list(range(8, 16)))) if SEP else (teacher(HF, list(range(16))), teacher(HS, list(range(16))))
xs = torch.randn(20000, IN, generator=g)
sf, ss = tf(xs).std(0).mean(), ts(xs).std(0).mean()
target = lambda x: torch.cat([A * tf(x) / sf, SLOWAMP * ts(x) / ss], 1)


def mlp(i, h, o):
    L = [nn.Linear(i, h), nn.GELU()]
    for _ in range(DEPTH - 1):
        L += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*L, nn.Linear(h, o))


def nparams(i, h, o):
    return i * h + h + (DEPTH - 1) * (h * h + h) + h * o + o


if NET == "split":
    nets = [mlp(IN, W, O), mlp(IN, W, O)]
    fwd = lambda x: torch.cat([nets[0](x), nets[1](x)], 1)
else:
    budget = 2 * nparams(IN, W, O)
    ws = max(h for h in range(1, 4 * W) if nparams(IN, h, 2 * O) <= budget)
    nets = [mlp(IN, ws, 2 * O)]
    fwd = nets[0]
params = [p for n in nets for p in n.parameters()]
N = sum(p.numel() for p in params)


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7); t = X.shape[0] > X.shape[1]
    if t: X = X.T
    for _ in range(steps):
        Am = X @ X.T; X = a * X + (b * Am + c * Am @ Am) @ X
    return X.T if t else X


class Muon(torch.optim.Optimizer):
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr))
    @torch.no_grad()
    def step(s):
        for gr in s.param_groups:
            for p in gr["params"]:
                st = s.state[p]; m = st.setdefault("m", torch.zeros_like(p)); m.mul_(0.95).add_(p.grad)
                u = newton_schulz(p.grad + 0.95 * m); p.add_(u, alpha=-gr["lr"] * max(1, p.shape[0] / p.shape[1]) ** 0.5)


if OPT == "muon":
    hid = [m.weight for n in nets for m in n if isinstance(m, nn.Linear) and m.in_features == m.out_features]
    ids = {id(p) for p in hid}
    opts = [Muon(hid, float(E("MUONLR", 0.02))), torch.optim.Adam([p for p in params if id(p) not in ids], lr=LR)]
else:
    opts = [torch.optim.Adam(params, lr=LR)]
for o in opts:
    for pg in o.param_groups: pg["base"] = pg["lr"]
gen = torch.Generator().manual_seed(1)
def noisy(y, gg):
    y = y.clone(); y[:, O:] += SNOISE * torch.randn(len(y), O, generator=gg); return y
xt = torch.randn(8192, IN, generator=torch.Generator().manual_seed(2)); yt = noisy(target(xt), torch.Generator().manual_seed(3))
var = yt.var(0)
if NTRAIN:
    XTR = torch.randn(NTRAIN, IN, generator=torch.Generator().manual_seed(4)); YTR = noisy(target(XTR), torch.Generator().manual_seed(5))
t0 = time.time()
for s in range(STEPS):
    if NTRAIN:
        i = torch.randint(0, NTRAIN, (BATCH,), generator=gen); x, y = XTR[i], YTR[i]
    else:
        x = torch.randn(BATCH, IN, generator=gen); y = noisy(target(x), gen)
    loss = ((fwd(x) - y) ** 2).mean()
    for o in opts: o.zero_grad()
    loss.backward()
    for o in opts:
        for pg in o.param_groups: pg["lr"] = pg["base"] * 0.5 * (1 + np.cos(np.pi * s / STEPS))
        o.step()
with torch.no_grad():
    e = ((fwd(xt) - yt) ** 2).mean(0) / var
cv = [float(e[:O].mean()), float(e[O:].mean())]
rec = dict(net=NET, w=W, N=N, steps=STEPS, a=A, hf=HF, hs=HS, depth=DEPTH, lr=LR, batch=BATCH, opt=OPT, slowamp=SLOWAMP, snoise=SNOISE, sep=SEP, ntrain=NTRAIN,
           muonlr=float(E("MUONLR", 0.02)) if OPT == "muon" else None, tag=E("TAG", ""), cv=cv,
           mse=[float(((fwd(xt) - yt) ** 2)[:, :O].mean()), float(((fwd(xt) - yt) ** 2)[:, O:].mean())] if False else None,
           var=[float(var[:O].mean()), float(var[O:].mean())], sec=round(time.time() - t0, 1))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher.jsonl"), "a").write(json.dumps(rec) + "\n")
print(json.dumps(rec))
