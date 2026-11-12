"""Fixed-slowdown model: shared error at S steps = own-network error at S / F steps, F measured once at 6k steps.
Predicts the shared error at other lengths from the own curve.   python3 slowdown.py"""
import json, numpy as np
R = [json.loads(l) for l in open("drive.jsonl")]
def base_ok(r, world):
    return ("carscale" in r and r.get("decay") and r.get("world", "drive") == world and r.get("loss") != "l1" and r.get("norm") == "none"
            and r.get("lr") == 1e-3 and r.get("batch") == 256 and r.get("depth", 2) == 2 and r.get("nclip", 1000) == 1000 and r["arch"] == "mlp"
            and r.get("opt", "adam") == "adam" and r.get("lw") in (None, [1.0, 1.0]))
def own_curve(world, size, slow, tags):
    return sorted({r["steps"]: r["cv"][slow] for r in R if base_ok(r, world) and r["net"] == "split" and r["size"] == size and r.get("tag") in tags}.items())
def shared(world, size, slow, a, st_, tags):
    return {r["steps"]: r["cv"][slow] for r in R if base_ok(r, world) and r["net"] == "single" and r["size"] == size and r["carscale"] == a
            and r.get("scalething", 0) == st_ and r.get("tag") in tags}
def own_at(c, S):
    s = np.log([x for x, _ in c]); e = np.log([y for _, y in c]); return float(np.exp(np.interp(np.log(S), s, e)))
def s_at(c, E):
    s = np.log([x for x, _ in c]); e = np.log([y for _, y in c]); return float(np.exp(np.interp(-np.log(E), -e, s)))
T = ("steps", "", None, "fine")
for world, size, slow, a, st_, tags in [("drive", "s", 1, 4.0, 0, T), ("drive", "m", 1, 1.0, 0, T), ("drive", "m", 1, 2.0, 0, T), ("drive", "m", 1, 4.0, 0, T),
                                        ("drive", "ml", 1, 4.0, 0, T), ("arm", "m", 0, 4.0, 1, ("armlen",))]:
    c = own_curve(world, size, slow, tags); sh = shared(world, size, slow, a, st_, tags)
    if 6000 not in sh: print(world, size, a, "no 6k"); continue
    F = 6000 / s_at(c, sh[6000])
    out = []
    for S in sorted(sh):
        if S == 6000: continue
        pred = own_at(c, S / F)
        out.append("%dk: measured %.3f predicted %.3f (x%.2f)" % (S // 1000, sh[S], pred, sh[S] / pred))
    print("%-5s %-2s gap x%-3g F %.2f | own %s | %s" % (world, size, a, F, " ".join("%dk %.3f" % (s // 1000 if s >= 1000 else s, e) for s, e in c[-4:]), "; ".join(out)))
