"""In-silico pilot for 'The Cellular Data Center'.

Compares, at equal size N, four substrates that feed the SAME trained part (a ridge-regression linear readout):
  input-only   : no reservoir, the readout sees only the current input (floor)
  ESN          : random rate-based echo state network (silicon reservoir baseline)
  LIF          : frozen spiking reservoir (leaky integrate-and-fire, 80/20 E/I, random 2D distance-dependent wiring, noise)
  LIF+STDP     : the same spiking reservoir after an unsupervised STDP exposure phase, then frozen

Tasks: linear memory capacity (Jaeger 2001), Mackey-Glass (tau=17) h-step prediction, next-character prediction (27 symbols).
Everything is numpy, deterministic per seed. Usage: python pilot.py [--quick]
"""
import json, sys, time, numpy as np

QUICK = '--quick' in sys.argv
N = 500
SEEDS = [0, 1, 2] if QUICK else [0, 1, 2, 3, 4]
WIN = 15            # ms of simulated spiking time per input symbol / sample
LAMBDAS = [1e-6, 1e-4, 1e-2, 1e0, 1e2]


# ----------------------------------------------------------------- readout
def ridge(X, Y, lam):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ Y)


def predict(X, W):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def zs(Xtr, *others):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    return [(Xtr - m) / s] + [(o - m) / s for o in others]


# ----------------------------------------------------------------- substrates
class ESN:
    def __init__(self, n_in, seed, rho=0.9, in_scale=0.5, leak=1.0, density=0.1):
        r = np.random.default_rng(1000 + seed)
        W = r.normal(0, 1, (N, N)) * (r.random((N, N)) < density)
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
        self.W, self.Win = W, r.uniform(-in_scale, in_scale, (N, n_in))
        self.b, self.leak = r.uniform(-.1, .1, N), leak

    def run(self, U):
        x = np.zeros(N); out = np.empty((len(U), N))
        for t, u in enumerate(U):
            x = (1 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.Win @ u + self.b)
            out[t] = x
        return out


class LIF:
    """Current-based LIF network, dt = 1 ms. Dimensionless voltage: rest 0, threshold 1, reset 0."""

    def __init__(self, n_in, seed, w_e=0.08, g=4.5, in_gain=0.6, sigma=0.15, mu=0.9, noise=0.25, in_frac=0.25, tau_syn=5., topo='2d', ring_w=0.5):
        r = np.random.default_rng(2000 + seed)
        self.r = np.random.default_rng(3000 + seed)
        nE = int(0.8 * N); self.nE = nE
        pos = r.random((N, 2))                              # neurons on a unit "dish"
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        P = 0.35 * np.exp(-d2 / (2 * sigma ** 2)); np.fill_diagonal(P, 0)
        C = r.random((N, N)) < P                            # C[i, j]: j -> i
        ring = np.zeros((N, N))
        if topo == 'ring':
            # 'memory ring': excitatory cells sit on a closed loop of 40 sectors; each sector projects to the next one
            sec = np.arange(nE) * 40 // nE
            fwd = (sec[:, None] - sec[None, :]) % 40 == 1
            ring[:nE, :nE] = ring_w * fwd * (r.random((nE, nE)) < 0.5) * r.uniform(.5, 1.5, (nE, nE))
        W = np.zeros((N, N))
        W[:, :nE] = w_e * C[:, :nE] * r.uniform(.5, 1.5, (N, nE))
        W[:, nE:] = -g * w_e * C[:, nE:] * r.uniform(.5, 1.5, (N, N - nE))
        W[:nE, :nE] += ring[:nE, :nE]
        C = C | (ring > 0)
        self.W, self.C = W, C
        self.topo = topo
        # "electrodes": each input channel drives a random footprint of excitatory neurons
        Win = np.zeros((N, n_in))
        for k in range(n_in):
            pool = np.arange(nE // 40) if topo == 'ring' and n_in == 1 else np.arange(nE)
            idx = r.choice(pool, min(len(pool), int(in_frac * nE / max(1, n_in ** .5))), replace=False)
            Win[idx, k] = r.uniform(.5, 1.5, len(idx)) * (1 if n_in > 1 else r.choice([-1, 1], len(idx)))
        self.Win, self.in_gain, self.mu, self.noise = Win * in_gain, in_gain, mu, noise
        self.ts, self.tm, self.tr = np.exp(-1 / tau_syn), 1 / 20., 2

    def run(self, U, stdp=False, A=0.004):
        v = self.r.random(N) * .5; I = np.zeros(N); ref = np.zeros(N); tr = np.zeros(N)
        pre = np.zeros(N); post = np.zeros(N)
        out = np.empty((len(U), N)); dtr = np.exp(-1 / 20.); dpl = np.exp(-1 / 20.)
        W, nE = self.W, self.nE
        mask = self.C[:nE, :nE] if stdp else None
        wmax = 3 * W[:nE, :nE].max() if stdp else None
        for t, u in enumerate(U):
            drive = self.Win @ u
            for _ in range(WIN):
                ext = self.mu + drive + self.noise * self.r.standard_normal(N)
                v += self.tm * (-v + ext + I)
                v[ref > 0] = 0; ref -= 1
                s = v >= 1.0
                v[s] = 0; ref[s] = self.tr
                sf = s.astype(float)
                I = I * self.ts + W @ sf
                tr = tr * dtr + sf
                if stdp:
                    pre = pre * dpl + sf; post = post * dpl + sf
                    if s[:nE].any():
                        sE = sf[:nE]
                        dW = A * (np.outer(sE, pre[:nE]) - 1.05 * np.outer(post[:nE], sE))
                        W[:nE, :nE] = np.clip(W[:nE, :nE] + dW * mask, 0, wmax)
            out[t] = tr
        return out

    def rate(self, X):
        return X.mean() / (20e-3)  # approx. spikes/s from 20 ms trace


# ----------------------------------------------------------------- tasks
def mackey_glass(n, tau=17, dt=0.1, sub=10, seed=0):
    r = np.random.default_rng(seed); hist = int(tau / dt)
    x = list(1.2 + 0.1 * r.standard_normal(hist + 1))
    for _ in range(n * sub + 3000):
        xt, xd = x[-1], x[-hist - 1]
        x.append(xt + dt * (0.2 * xd / (1 + xd ** 10) - 0.1 * xt))
    x = np.array(x[3000::sub])[:n]
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def task_mc(make, seed, stdp=False):
    r = np.random.default_rng(seed); T, wash, K = 4200, 200, 80
    u = r.uniform(-1, 1, T)
    sub = make(1, seed)
    if stdp: sub.run(r.uniform(-1, 1, (1500, 1)), stdp=True)
    X = sub.run(u[:, None]) if sub is not None else u[:, None]
    X, u = X[wash:], u[wash:]
    tr, va = slice(K, 2600), slice(2600, 3200); te = slice(3200, None)
    Xtr, Xva, Xte = zs(X[tr], X[va], X[te])
    mc = 0
    for k in range(1, K + 1):
        y = np.roll(u, k)
        best = None
        for lam in LAMBDAS:
            Wt = ridge(Xtr, y[tr, None], lam)
            sc = np.corrcoef(predict(Xva, Wt)[:, 0], y[va])[0, 1] ** 2
            if best is None or sc > best[0]: best = (sc, lam)
        Wt = ridge(np.vstack([Xtr, Xva]), np.concatenate([y[tr], y[va]])[:, None], best[1])
        c = np.corrcoef(predict(Xte, Wt)[:, 0], y[te])[0, 1]
        mc += 0 if np.isnan(c) else c ** 2
    return mc, X


def task_mg(make, seed, stdp=False, h=5):
    s = mackey_glass(4300, seed=seed)
    u, y = s[:-h], s[h:]
    sub = make(1, seed)
    if stdp: sub.run(np.random.default_rng(seed).uniform(-1, 1, (1500, 1)), stdp=True)
    X = sub.run(u[:, None]) if sub is not None else u[:, None]
    wash = 200; X, y = X[wash:], y[wash:]
    tr, va, te = slice(0, 2600), slice(2600, 3100), slice(3100, None)
    Xtr, Xva, Xte = zs(X[tr], X[va], X[te])
    best = min(((np.sqrt(np.mean((predict(Xva, ridge(Xtr, y[tr, None], l))[:, 0] - y[va]) ** 2)) / np.std(y[va]), l) for l in LAMBDAS))
    Wt = ridge(np.vstack([Xtr, Xva]), np.concatenate([y[tr], y[va]])[:, None], best[1])
    return float(np.sqrt(np.mean((predict(Xte, Wt)[:, 0] - y[te]) ** 2)) / np.std(y[te]))


VOC = 'abcdefghijklmnopqrstuvwxyz '
def corpus():
    raw = open('tinyshakespeare.txt').read().lower()
    txt = ''.join(c if c in VOC else ' ' for c in raw)
    txt = ' '.join(txt.split())
    return np.array([VOC.index(c) for c in txt[:40000]])


def task_char(make, seed, stdp=False):
    ids = corpus(); n = len(ids); V = len(VOC)
    U = np.eye(V)[ids]
    sub = make(V, seed)
    if stdp: sub.run(U[:1500], stdp=True)
    X = sub.run(U[:-1]) if sub is not None else U[:-1]
    Y = U[1:]; wash = 100
    X, Y = X[wash:], Y[wash:]
    tr, va, te = slice(0, 24000), slice(24000, 29000), slice(29000, None)
    Xtr, Xva, Xte = zs(X[tr], X[va], X[te])
    best = max(((np.mean(np.argmax(predict(Xva, ridge(Xtr, Y[tr], l)), 1) == np.argmax(Y[va], 1)), l) for l in LAMBDAS))
    Wt = ridge(np.vstack([Xtr, Xva]), np.vstack([Y[tr], Y[va]]), best[1])
    return float(np.mean(np.argmax(predict(Xte, Wt), 1) == np.argmax(Y[te], 1)))


def ngram_refs():
    ids = corpus(); tr, te = ids[100:29100], ids[29100:]
    uni = np.bincount(tr, minlength=27).argmax()
    big = np.zeros((27, 27)); np.add.at(big, (tr[:-1], tr[1:]), 1)
    tri = {}
    for a, b, c in zip(tr[:-2], tr[1:-1], tr[2:]): tri.setdefault((a, b), np.zeros(27))[c] += 1
    acc_u = np.mean(te[1:] == uni)
    acc_b = np.mean(big[te[:-1]].argmax(1) == te[1:])
    acc_t = np.mean([(tri[(a, b)].argmax() if (a, b) in tri else big[b].argmax()) == c for a, b, c in zip(te[:-2], te[1:-1], te[2:])])
    return dict(unigram=float(acc_u), bigram=float(acc_b), trigram=float(acc_t))


# settings chosen on tuning seed 99 (tune.py): ESN rho=1.05, input scale 0.1; LIF input gain 1.2, noise 0.1
SUBS = {
    'input-only': lambda n_in, seed: None,
    'ESN': lambda n_in, seed: ESN(n_in, seed, rho=1.05, in_scale=0.1),
    'LIF': lambda n_in, seed: LIF(n_in, seed, in_gain=1.2, noise=0.1),
    'LIF-slow': lambda n_in, seed: LIF(n_in, seed, in_gain=1.2, noise=0.1, tau_syn=50.),
    'LIF+STDP': lambda n_in, seed: LIF(n_in, seed, in_gain=1.2, noise=0.1),
}

if __name__ == '__main__':
    res = {k: {'MC': [], 'MG_NRMSE': [], 'char_acc': []} for k in SUBS}
    rates = []
    t0 = time.time()
    for seed in SEEDS:
        for name, mk in SUBS.items():
            st = name == 'LIF+STDP'
            mc, X = task_mc(mk, seed, st)
            res[name]['MC'].append(float(mc))
            if name.startswith('LIF'): rates.append([name, seed, float(X.mean() / 0.020)])
            res[name]['MG_NRMSE'].append(task_mg(mk, seed, st))
            res[name]['char_acc'].append(task_char(mk, seed, st))
            print(f'seed {seed} {name:10s} MC={mc:6.2f} MG={res[name]["MG_NRMSE"][-1]:.4f} char={res[name]["char_acc"][-1]:.4f}  [{time.time()-t0:.0f}s]', flush=True)
    out = {'N': N, 'seeds': SEEDS, 'win_ms': WIN, 'results': res, 'ngram': ngram_refs(), 'LIF_mean_rate_hz': rates}
    json.dump(out, open('results/pilot_results.json', 'w'), indent=1)
    print(json.dumps(out['ngram']), rates)
