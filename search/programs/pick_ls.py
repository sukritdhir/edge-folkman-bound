"""Local search (swaps, restarts) for the extra set X of a given size maximising g(X) at fixed weights (see pick_extra.py).
usage: python3 exp/pick_ls.py SPEC result.pkl beta t [rounds] [seed]"""
import sys, pickle, time
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np, pick_extra

spec, pkl, beta, t = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 20; seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
P = pick_extra.setup(spec, pkl, beta); n = P["n"]; W3 = 3 * P["W"]
rng = np.random.default_rng(seed); keep = np.nonzero(P["keep"])[0]
T = P["T"]; wt = P["wt"]
# vertex -> triangles
inc = [[] for _ in range(n)]
for i, (a, b, c) in enumerate(T): inc[a].append(i); inc[b].append(i); inc[c].append(i)
inc = [np.array(x, dtype=np.int64) for x in inc]

def marginal(Xs, Xl):
    """gain of adding each vertex to X"""
    base = -(3 * P["wv"] - P["Dv"]) + P["Epair"][:, Xl].sum(axis=1)
    cnt = Xs[T].sum(axis=1); sel = cnt == 2
    if sel.any():
        Ts = T[sel]; third = Ts[~Xs[Ts]]
        pen = np.zeros(n); np.add.at(pen, third, 3 * wt[sel]); base -= pen
    return base

best = None; t0 = time.time()
for rd in range(rounds):
    s = int(rng.choice(keep)); X, hist = pick_extra.greedy(P, t, [s])
    g = pick_extra.gval(P, X); improved = True
    while improved:
        improved = False
        for i in rng.permutation(t):
            Y = X[:i] + X[i + 1:]; Ys = np.zeros(n, bool); Ys[Y] = True
            m = marginal(Ys, Y); m[Ys] = -np.inf; m[~P["keep"]] = -np.inf
            v = int(np.argmax(m))
            if v != X[i]:
                gy = pick_extra.gval(P, Y + [v])
                if gy > g + 1e-12: X = Y + [v]; g = gy; improved = True
    if best is None or g > best[0]:
        best = (g, list(X)); e_int = int(P["A"][np.ix_(X, X)].sum() // 2)
        print("round %d (seed vertex %d): g/3W = %+.6f, internal edges %d, X = %s  [%.0fs]" % (rd, s, g / W3, e_int, ",".join(str(x) for x in X), time.time() - t0)); sys.stdout.flush()
pickle.dump(best, open("scan/pick_ls_%d.pkl" % t, "wb"))
