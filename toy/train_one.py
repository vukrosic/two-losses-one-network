"""Train one model (under 10 s), report the halves probe and missed change, save its latents.
    python3 train_one.py REPO tag partner kind steps lr [wb=10] [per_dot=1] [sigreg=..] [latent=..]
wb = weight of half B's prediction error in the loss (1 = the repo's loss)."""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(2)
repo, tag, partner, kind, steps, lr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6])
opt, wb, data, H = {}, 1.0, "mix", None
for kv in sys.argv[7:]:
    key, v = kv.split("=")
    if key == "wb": wb = float(v); continue
    if key == "data": data = v; continue
    if key == "h": H = int(v); continue
    opt[key] = bool(int(v)) if key in ("per_dot", "action", "gray") else int(v) if key in ("latent", "img") else float(v)
sys.path.insert(0, repo)
import model as Mo, run as R, world as W
t0 = time.time()
c, k = R.TIERS["7s"], 3
O, U = W.collect(partner, c["traj"], c["T"], 0, R.SEEK[data])
Oe, Ue = W.collect(partner, 32, c["T"], 200, R.SEEK[data])
model = Mo.make(kind, partner, k, H or c["h"], seed=0, **opt)
if wb != 1.0:
    def losses(model, O, U):
        k, D = model.k, model.D
        z = model.embed(model.pictures(O))
        U = torch.from_numpy(U).to(z.device)
        P, Tg = [], []
        for j in range(k - 1, U.shape[1]):
            zw = z[:, j - k + 1:j + 1].flatten(1)
            P.append(zw[:, -D:] + model.delta(zw, U[:, j]))
            Tg.append(z[:, j + 1])
        P, Tg = torch.stack(P, 1), torch.stack(Tg, 1)
        pred = (F.mse_loss(P[..., :D // 2], Tg[..., :D // 2]) + wb * F.mse_loss(P[..., D // 2:], Tg[..., D // 2:])) / 2
        return pred, model.sig(z.transpose(0, 1)), torch.zeros(())
    Mo.losses = losses
Mo.train(model, O, U, steps, lr, 0, None, device=Mo.pick_device())
ttrain = time.time() - t0
h = Mo.halves(model, O, Oe)
Y, Ye = O.reshape(-1, W.ODIM), Oe.reshape(-1, W.ODIM)
with torch.no_grad():
    Z, Ze = model.embed(model.pictures(Y)), model.embed(model.pictures(Ye))
    z, Ut, D = model.embed(model.pictures(Oe)), torch.from_numpy(Ue), model.D
    num, den = torch.zeros(2), torch.zeros(2)
    for j in range(k - 1, Ut.shape[1]):
        zw = z[:, j - k + 1:j + 1].flatten(1)
        pred = zw[:, -D:] + model.delta(zw, Ut[:, j])
        for i, sl in enumerate((slice(0, D // 2), slice(D // 2, D))):
            num[i] += F.mse_loss(pred[:, sl], z[:, j + 1, sl], reduction="sum")
            den[i] += F.mse_loss(zw[:, -D:][:, sl], z[:, j + 1, sl], reduction="sum")
    miss = (num / den.clamp_min(1e-12)).tolist()
guess = [float(np.linalg.norm(Ye[:, s] - Y[:, s].mean(0), axis=-1).mean()) for s in (slice(0, 2), slice(2, 4))]
torch.save(model.state_dict(), f"{tag}.pt")
np.savez(f"{tag}.npz", Z=Z.numpy(), Y=Y, Ze=Ze.numpy(), Ye=Ye)
r = dict(tag=tag, partner=partner, kind=kind, steps=steps, lr=lr, opt=opt, wb=wb, halves=h, guess=guess, missA=miss[0], missB=miss[1],
         spread=float(Ze.std(0).mean()), train_s=round(ttrain, 2), seconds=round(time.time() - t0, 2))
print(f"{tag:14s} | robot A {h['me_from_A']:.3f} B {h['me_from_B']:.3f} (guess {guess[0]:.3f}) | other A {h['other_from_A']:.3f} B {h['other_from_B']:.3f} (guess {guess[1]:.3f}) | miss A {miss[0]:.2f} B {miss[1]:.2f} | spread {r['spread']:.2f} | {r['seconds']} s")
open("r.jsonl", "a").write(json.dumps(r) + "\n")
