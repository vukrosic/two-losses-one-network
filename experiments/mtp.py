"""Multi-token prediction test: do extra future-token heads hurt next-token prediction, and does the loss-ratio law
predict by how much?

    python3 mtp.py own K d [steps]      one model with one output head: predicts the token K places ahead
    python3 mtp.py shared K d [steps]   one model with heads for 1..K places ahead; loss = CE_1 + W * mean(CE_2..CE_K)
                                        (the weighting DeepSeek-V3 uses for its multi-token prediction loss)

Data: an order-ORDER Markov chain over V tokens with random Dirichlet rows. The true distribution of every future
token is known exactly, so each head's score is its excess loss = cross-entropy - true entropy (the part a model can
still learn), computed exactly over a fixed evaluation set. Fresh sequences every step (no repeated data). The chain,
the training data stream and the starting weights are identical in every run; only the heads and widths differ.

env: V (64), ORDER (2), ALPHA (0.5, Dirichlet concentration; smaller = more predictable), ZIPF (1: tokens drawn with
     1/rank frequencies, as in text), W (1.0, weight on the future heads), LAYERS (2), NHEAD (4), SEQ (64), BATCH (32),
     LR (3e-3, Adam, warmup WARM=50 then cosine to 0), HEADBLOCK (0; 1 = each head gets its own transformer block
     before its output layer, as in Gloeckle et al. 2024), KMAX (4, longest offset the data supports), THREADS (2), TAG
Model: pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied output layer per head.
Appends one JSON line to mtp.jsonl: per-head excess loss, cross-entropy and entropy at 1/8, 1/4, 1/2 and all of training.
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get
torch.set_num_threads(int(E("THREADS", 2)))
NET, K, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
V, ORDER, ALPHA, ZIPF = int(E("V", 64)), int(E("ORDER", 2)), float(E("ALPHA", 0.5)), E("ZIPF", "1") == "1"
W, LAYERS, NH, SEQ, BATCH = float(E("W", 1.0)), int(E("LAYERS", 2)), int(E("NHEAD", 4)), int(E("SEQ", 64)), int(E("BATCH", 32))
LR, WARM, HEADBLOCK, KMAX = float(E("LR", 3e-3)), int(E("WARM", 50)), E("HEADBLOCK", "0") == "1", int(E("KMAX", 4))
assert NET in ("own", "shared") and 1 <= K <= KMAX and D % NH == 0 and (D // NH) % 2 == 0
OFFS = list(range(1, K + 1)) if NET == "shared" else [K]
S, BURN = V ** ORDER, ORDER + 16

# ---- the world: a fixed Markov chain and the exact distribution of the token k places ahead, for every state ----
base = 1.0 / np.arange(1, V + 1) if ZIPF else np.ones(V)
P1 = np.random.default_rng(0).dirichlet(ALPHA * base / base.sum() * V, size=S) + 1e-9
P1 = torch.tensor(P1 / P1.sum(1, keepdims=True), dtype=torch.float32)          # state (last ORDER tokens) -> next token
nxt = (torch.arange(S)[:, None] * V + torch.arange(V)[None, :]) % S              # state after emitting token c
PK = {1: P1}
for k in range(2, KMAX + 1):
    out = torch.empty(S, V)
    for i in range(0, S, 256):
        out[i:i + 256] = torch.einsum("sc,scv->sv", P1[i:i + 256], PK[k - 1][nxt[i:i + 256]])
    PK[k] = out


def sample(n, gen):
    """n sequences of SEQ + KMAX tokens and the chain state after each token"""
    st, toks, sts = torch.randint(0, S, (n,), generator=gen), [], []
    for i in range(BURN + SEQ + KMAX):
        c = torch.multinomial(P1[st], 1, generator=gen).squeeze(1)
        st = (st * V + c) % S
        if i >= BURN: toks.append(c); sts.append(st)
    return torch.stack(toks, 1), torch.stack(sts, 1)


# ---- the model ----
def rope(x, base=10000):
    t, d = x.shape[-2], x.shape[-1]
    ang = torch.outer(torch.arange(t).float(), 1.0 / base ** (torch.arange(0, d, 2).float() / d))
    c, s = ang.cos(), ang.sin(); a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * c - b * s, b * c + a * s], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d // NH), nn.RMSNorm(d // NH)
        s.qkv, s.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        h = 8 * d // 3
        s.gate, s.up, s.down = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(s, x):
        b, t, d = x.shape
        q, k, v = s.qkv(s.n1(x)).view(b, t, 3, NH, d // NH).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(s.qn(q)), rope(s.kn(k)), v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(b, t, d))
        h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.blocks = nn.Embedding(V, D), nn.ModuleList(Block(D) for _ in range(LAYERS))
        s.heads = nn.ModuleDict({str(k): nn.Sequential(*([Block(D)] if HEADBLOCK else []), nn.RMSNorm(D),
                                                        nn.Linear(D, V, bias=False)) for k in OFFS})

    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return {k: s.heads[str(k)](h) for k in OFFS}


torch.manual_seed(3)
model = LM()
N = sum(p.numel() for p in model.parameters())
N_trunk = N - sum(p.numel() for p in model.heads.parameters())
opt = torch.optim.Adam(model.parameters(), lr=LR)
gen = torch.Generator().manual_seed(1)
XE, SE = sample(256, torch.Generator().manual_seed(2))


@torch.no_grad()
def evaluate():
    """exact per-head scores; positions before ORDER-1 are skipped (the model cannot see the full state there)"""
    lo, out = model(XE[:, :SEQ]), {}
    for k in OFFS:
        p = PK[k][SE[:, ORDER - 1:SEQ]]
        lq = F.log_softmax(lo[k][:, ORDER - 1:], -1)
        ce, H = float(-(p * lq).sum(-1).mean()), float(-torch.xlogy(p, p).sum(-1).mean())
        out[k] = dict(excess=ce - H, ce=ce, H=H)
    return out


marks = {max(1, STEPS // 8): "1/8", max(1, STEPS // 4): "1/4", max(1, STEPS // 2): "1/2", STEPS: "1"}
curve, t0 = {}, time.time()
for s in range(STEPS):
    toks, _ = sample(BATCH, gen)
    lo = model(toks[:, :SEQ])
    ce = {k: F.cross_entropy(lo[k].flatten(0, 1), toks[:, k:k + SEQ].flatten()) for k in OFFS}
    loss = ce[K] if NET == "own" else ce[1] + (W * sum(ce[k] for k in OFFS[1:]) / (K - 1) if K > 1 else 0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for pg in opt.param_groups: pg["lr"] = LR * min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * s / STEPS))
    opt.step()
    if s + 1 in marks: curve[marks[s + 1]] = evaluate()

rec = dict(net=NET, K=K, offs=OFFS, d=D, N=N, N_trunk=N_trunk, steps=STEPS, W=W if NET == "shared" else None, V=V,
           order=ORDER, alpha=ALPHA, zipf=ZIPF, layers=LAYERS, headblock=HEADBLOCK, lr=LR, batch=BATCH, seq=SEQ,
           tag=E("TAG", ""), final=curve["1"], curve=curve, sec=round(time.time() - t0, 1))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp.jsonl"), "a").write(json.dumps(rec) + "\n")
print(json.dumps(dict(net=NET, K=K, d=D, N=N, W=rec["W"], steps=STEPS, sec=rec["sec"],
                      excess={k: round(v["excess"], 4) for k, v in curve["1"].items()})))
