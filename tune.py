"""Small, symmetric hyper-parameter search on a held-out tuning seed (99), memory-capacity task only."""
import json, itertools, pilot as P
out = {'ESN': [], 'LIF': []}
for rho, isc in itertools.product([0.8, 0.95, 1.05], [0.1, 0.5, 1.0]):
    mc, _ = P.task_mc(lambda n, s: P.ESN(n, s, rho=rho, in_scale=isc), 99); out['ESN'].append([rho, isc, round(mc, 2)])
for gain, noise in itertools.product([0.3, 0.6, 1.2], [0.1, 0.25, 0.5]):
    mc, _ = P.task_mc(lambda n, s: P.LIF(n, s, in_gain=gain, noise=noise), 99); out['LIF'].append([gain, noise, round(mc, 2)])
for k, v in out.items(): print(k, sorted(v, key=lambda r: -r[2])[:4])
json.dump(out, open('results/tune_results.json', 'w'), indent=1)
