"""Capacity-split prediction of the shared-network penalty P.

Given each thing's own error curve E_i(w) (error in loss units of a separate network with w weights),
the shared network of W weights is modelled as giving w_i weights to thing i with sum w_i = W, chosen to minimise
sum E_i(w_i). Curves are interpolated linearly in log(error) vs log(weights) and extended flat at the ends.
P_i = E_i(w_i*) / E_i(W / M), the predicted error in one network / error with its own network.

    from capsplit import predict
    predict({"car": [(w, err), ...], "ped": [...]}, W) -> {"car": (w*, E*, P), ...}
"""
import numpy as np


def curve(pts):
    pts = sorted(pts)
    lw, le = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return lambda w: float(np.exp(np.interp(np.log(w), lw, le)))


def predict(curves, W, grid=400):
    names = list(curves)
    fs = [curve(curves[n]) for n in names]
    if len(names) != 2:
        raise ValueError("two things only")
    share = np.linspace(0.01, 0.99, grid)
    tot = [fs[0](s * W) + fs[1]((1 - s) * W) for s in share]
    s = share[int(np.argmin(tot))]
    ws = [s * W, (1 - s) * W]
    return {n: (w, f(w), f(w) / f(W / 2)) for n, f, w in zip(names, fs, ws)}
