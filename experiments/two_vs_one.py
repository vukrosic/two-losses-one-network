"""Env SW / RW = weight of the stranger / robot half in the loss (default 1). Env NET=split|single runs one.
Env DECAY=1 cosine learning-rate decay to 0 over the run. Env NORM=1 divides each number's error by its typical squared one-step change on the training clips.
2 vs 1 predictor on one frozen encoder. Loads tag.pt (a train_one.py per_dot model), trains a fresh split predictor
(robot net width 32 + stranger net width 64, 16,160 weights) and a fresh single net (width 77, 16,125 weights)
on the same latents; reports held-out robot / stranger miss at checkpoints.
    python3 two_vs_one.py REPO tag partner kind lr steps1,steps2,.. [ntraj=48] [robot,stranger,single widths=32,64,77] [smove=1: stranger net also sees the move]   (kind = how tag.pt was trained)"""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(2)
repo, tag, partner, kind, lr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
marks = [int(x) for x in sys.argv[6].split(",")]
ntraj = int(sys.argv[7]) if len(sys.argv) > 7 else 48
smove = len(sys.argv) > 9 and sys.argv[9] == "smove=1"
wa, wb_, ws = [int(x) for x in sys.argv[8].split(",")] if len(sys.argv) > 8 else (32, 64, 77)
sys.path.insert(0, repo)
import model as Mo, run as R, world as W
t0 = time.time()
c, k = R.TIERS["7s"], 3
if os.environ.get("PSPEED"):          # stranger's speed per step (world default 0.03; the robot's is 0.04)
    W.PSPEED = float(os.environ["PSPEED"])
A = int(os.environ.get("AHEAD", 1))   # predict A steps ahead, given the A moves in between
if os.environ.get("RSPEED"):          # the robot's speed per step (world default 0.04)
    W.SPEED = float(os.environ["RSPEED"])
MODE = os.environ.get("MODE")         # with partner=chase, swap the stranger's rule: flee (moves away from the robot), orbit (circles the robot)
if MODE:
    _pt = W.partner_target
    def _mode_target(p, s, g):
        if p != "chase":
            return _pt(p, s, g)
        me, ot = s[:, 0:2], s[:, 2:4]
        if MODE == "flee":
            return ot - W._unit(me - ot)
        if MODE == "orbit":
            return me + 0.15 * np.stack([np.cos(s[:, 6]), np.sin(s[:, 6])], 1)
        raise ValueError(MODE)
    W.partner_target = _mode_target
NOISE = float(os.environ.get("NOISE", 0))   # random push on the stranger each step (std, table units)
if NOISE:
    _st, _nrng = W.step, np.random.default_rng(5)
    def _noisy_step(p, s, a, g):
        s = _st(p, s, a, g)
        s[:, 2:4] = np.clip(s[:, 2:4] + _nrng.normal(0, NOISE, (len(s), 2)), W.R, 1 - W.R)
        return s
    W.step = _noisy_step
ORACLE = os.environ.get("ORACLE") == "1"   # every net also gets the true state at time t: both positions, the patrol/orbit angle, the switch mode
_rec, _st2 = [], W.step
def _rec_step(p, s, a, g):
    s = _st2(p, s, a, g)
    _rec.append(s[:, 6:8].copy())
    return s
W.step = _rec_step
O, U = W.collect(partner, ntraj, c["T"], 0, R.SEEK["mix"])
S = np.stack(_rec, 1); _rec.clear()
Oe, Ue = W.collect(partner, 32, c["T"], 200, R.SEEK["mix"])
Se = np.stack(_rec, 1); _rec.clear()
m = Mo.make(kind, partner, k, c["h"], seed=0, per_dot=True, match=(kind == "joint"))
m.load_state_dict(torch.load(f"{tag}.pt"))
D, H = m.D, m.D // 2
E = 7 if ORACLE else 0
def pairs(O, U, S):
    with torch.no_grad():
        z = m.embed(m.pictures(O))
    U = torch.from_numpy(U); js = range(k - 1, U.shape[1] - A + 1)
    def extra(j):   # true state at time j (S[:, j-1] is the state after step j)
        if not ORACLE:
            return torch.zeros(len(O), 0)
        th, md = S[:, j - 1, 0], S[:, j - 1, 1]
        return torch.from_numpy(np.concatenate([O[:, j, 0:4], np.cos(th)[:, None], np.sin(th)[:, None], md[:, None]], 1).astype(np.float32))
    X = torch.cat([torch.cat([z[:, j - k + 1:j + 1].flatten(1), U[:, j:j + A].flatten(1), extra(j)], -1) for j in js])
    return X, torch.cat([z[:, j] for j in js]), torch.cat([z[:, j + A] for j in js])
tr, te = pairs(O, U, S), pairs(Oe, Ue, Se)
LATSCALE = float(os.environ.get("LATSCALE", 1))   # measure one dot's latent half in LATSCALE x bigger units (targets only; inputs unchanged)
LATTHING = int(os.environ.get("LATTHING", 0))     # 0 = robot half, 1 = stranger half
if LATSCALE != 1:
    sl = slice(0, H) if LATTHING == 0 else slice(H, D)
    for t_ in (tr, te):
        t_[1][:, sl] *= LATSCALE; t_[2][:, sl] *= LATSCALE
_ch = (tr[2] - tr[1]).pow(2).mean(0)
print(f"typical change robot {_ch[:H].mean().sqrt():.3f} stranger {_ch[H:].mean().sqrt():.3f}")
norm = os.environ.get("NORM") == "1"
scale = (tr[2] - tr[1]).pow(2).mean(0) if norm else torch.ones(D)   # each number's typical squared one-step change
def nets(split):
    torch.manual_seed(0)
    so = smove or ORACLE   # the oracle stranger net sees everything
    ns = [Mo.mlp(k * D + 2 * A + E, wa, H), Mo.mlp(k * D + (2 * A + E) * so, wb_, H)] if split else [Mo.mlp(k * D + 2 * A + E, ws, D)]
    for n in ns:
        torch.nn.init.normal_(n[-1].weight, std=0.01); torch.nn.init.zeros_(n[-1].bias)
    f = (lambda X: torch.cat([ns[0](X), ns[1](X if so else X[:, :k * D])], -1)) if split else ns[0]
    return ns, f
def miss(f, X, cur, nxt):
    with torch.no_grad():
        e, b = (cur + f(X) - nxt).pow(2), (cur - nxt).pow(2)
        return [round(float(e[:, s].sum() / b[:, s].sum()), 3) for s in (slice(0, H), slice(H, D))]
out = {}
for split in {"split": (True,), "single": (False,)}.get(os.environ.get("NET"), (True, False)):
    ns, f = nets(split)
    ps = [p for n in ns for p in n.parameters()]
    o = torch.optim.SGD(ps, lr=lr, momentum=0.9) if os.environ.get("OPT") == "sgd" else torch.optim.Adam(ps, lr=lr)   # OPT=sgd: plain SGD with momentum
    X, cur, nxt = tr
    g = torch.Generator().manual_seed(0)
    for s in range(1, marks[-1] + 1):
        if os.environ.get("DECAY") == "1":   # cosine learning-rate decay from lr to 0 over the run
            for pg in o.param_groups:
                pg["lr"] = lr * 0.5 * (1 + np.cos(np.pi * (s - 1) / marks[-1]))
        j = torch.randint(0, len(X), (256,), generator=g)
        e = (cur[j] + f(X[j]) - nxt[j]).pow(2) / scale
        sw = float(os.environ.get("SW", 1))
        rw = float(os.environ.get("RW", 1))
        loss = (rw * e[:, :H].mean() + sw * e[:, H:].mean()) / 2
        o.zero_grad(); loss.backward(); o.step()
        if s in marks:
            out[("split" if split else "single", s)] = dict(test=miss(f, *te), train=miss(f, *tr))
for (name, s), v in out.items():
    print(f"{tag:10s} {partner:6s} {name:6s} {s:5d} steps | held-out miss robot {v['test'][0]:.2f} stranger {v['test'][1]:.2f} | train robot {v['train'][0]:.2f} stranger {v['train'][1]:.2f}")
print(f"{time.time() - t0:.1f} s")
with open("two_vs_one.jsonl", "a") as fh:
    for (name, s), v in out.items():
        fh.write(json.dumps(dict(tag=tag, partner=partner, ntraj=ntraj, widths=[wa, wb_, ws], smove=smove, sw=float(os.environ.get("SW", 1)), rw=float(os.environ.get("RW", 1)), norm=norm, enc=kind, lr=lr, pspeed=W.PSPEED, ahead=A, oracle=ORACLE, opt=os.environ.get("OPT", "adam"), latscale=LATSCALE, latthing=LATTHING, D=D, params=sum(p.numel() for p in nets(name == 'split')[0][0].parameters()) + (sum(p.numel() for p in nets(True)[0][1].parameters()) if name == 'split' else 0), rspeed=W.SPEED, mode=MODE, noise=NOISE, decay=os.environ.get("DECAY") == "1", ch_robot=float(_ch[:H].mean().sqrt()), ch_stranger=float(_ch[H:].mean().sqrt()), net=name, steps=s, **v)) + "\n")
