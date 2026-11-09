"""Single-seed probe of a naive 'memory ring' topology vs the 2D baseline (delay-wise linear memory r^2, k = 0..11)."""
import json, numpy as np, pilot as P
from probe import caps
cfgs = {'2D baseline': dict(), '2D, slow synapses (50 ms)': dict(tau_syn=50.),
        'ring, w=0.1': dict(topo='ring', ring_w=0.1), 'ring, w=0.2': dict(topo='ring', ring_w=0.2),
        'ring, w=0.5': dict(topo='ring', ring_w=0.5), 'ring, w=1.0': dict(topo='ring', ring_w=1.0)}
out = {}
for k, kw in cfgs.items():
    kw.update(in_gain=1.2, noise=0.1); c, r = caps(P.LIF(1, 0, **kw), K=12)
    out[k] = dict(r2=[float(x) for x in c], mc_k_ge_1=float(sum(c[1:])), rate_hz=float(r)); print(k, out[k]['mc_k_ge_1'], r)
json.dump(out, open('results/ring_probe.json', 'w'), indent=1)
