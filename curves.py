"""Delay-resolved linear memory r^2(k), seed 0, for the figure."""
import json, numpy as np, pilot as P
def curve(sub, K=40):
    r = np.random.default_rng(7); u = r.uniform(-1, 1, 3400)
    X = sub.run(u[:, None])[200:] if sub is not None else u[200:, None]; u = u[200:]
    tr, te = slice(K, 2400), slice(2400, None); Xtr, Xte = P.zs(X[tr], X[te]); out = []
    for k in range(0, K + 1):
        y = np.roll(u, k); W = P.ridge(Xtr, y[tr, None], 1e-2); c = np.corrcoef(P.predict(Xte, W)[:, 0], y[te])[0, 1]
        out.append(0.0 if np.isnan(c) else float(c ** 2))
    return out
res = {k: curve(P.SUBS[k](1, 0)) for k in ['ESN', 'LIF', 'LIF-slow']}
json.dump(res, open('results/curves.json', 'w'))
print({k: round(sum(v[1:]), 2) for k, v in res.items()})
