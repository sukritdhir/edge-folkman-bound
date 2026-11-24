"""Greedy choice of extra vertices X at fixed symmetric weights (w, D): maximise
   g(X) = 3 * (weight of the triangles avoiding X) - (sum of D over the edges avoiding X)       (certificate iff g >= 0
with A positive definite).  Marginal gain of adding v: -(3 w(v) - D(v)) + sum over u in X adjacent to v of (3 w(uv) - D_uv)
- 3 * (weight of the triangles with v and two vertices of X).
usage: python3 exp/pick_extra.py SPEC result.pkl beta tmax [seeds]      beta: the optimum (D of the file is scaled to budget 1)
"""
import sys, pickle
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np, vcost

def setup(spec, pkl, beta):
    A, perm, removed, G_, Sx, wt, De = vcost.load(spec, pkl)
    De = De * beta
    n = G_.n; W = wt.sum()
    wv = np.zeros(n); Dv = np.zeros(n)
    for j in range(3): np.add.at(wv, Sx.tri_verts[:, j], wt)
    for j in range(2): np.add.at(Dv, G_.ends[:, j], De)
    we = np.zeros(len(De))
    for j in range(3): np.add.at(we, Sx.tri_kept[:, j], wt)
    Epair = np.zeros((n, n)); u, v = G_.ends[:, 0], G_.ends[:, 1]
    Epair[u, v] = 3 * we - De; Epair[v, u] = 3 * we - De                  # gain of an internal edge
    # triangle weights by vertex triple
    tw = {}
    T = Sx.tri_verts
    return dict(n=n, W=W, wv=wv, Dv=Dv, Epair=Epair, T=T, wt=wt, keep=Sx.keep, g0=3 * W - De.sum(), A=A)

def gval(P, X):
    X = list(X); Xs = np.zeros(P["n"], bool); Xs[X] = True
    g = P["g0"] - (3 * P["wv"][X] - P["Dv"][X]).sum() + P["Epair"][np.ix_(X, X)].sum() / 2
    T = P["T"]; cnt = Xs[T].sum(axis=1)
    # inclusion-exclusion for triangles: a triangle with j vertices in X was subtracted j times, added back C(j,2) times; it must count once
    g -= 3 * P["wt"][cnt == 3].sum()
    return g

def greedy(P, tmax, start):
    n = P["n"]; X = list(start); Xs = np.zeros(n, bool); Xs[X] = True
    T = P["T"]; wt = P["wt"]
    hist = []
    while len(X) < tmax:
        base = -(3 * P["wv"] - P["Dv"]) + P["Epair"][:, X].sum(axis=1)
        # triangles with exactly two vertices in X: adding the third costs 3 w_T
        cnt = Xs[T].sum(axis=1); sel = cnt == 2
        if sel.any():
            Ts = T[sel]; third = Ts[~Xs[Ts]]
            pen = np.zeros(n); np.add.at(pen, third, 3 * wt[sel]); base -= pen
        base[Xs] = -np.inf; base[~P["keep"]] = -np.inf
        v = int(np.argmax(base)); X.append(v); Xs[v] = True
        hist.append((len(X), gval(P, X) / (3 * P["W"])))
    return X, hist

if __name__ == "__main__":
    spec, pkl, beta, tmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    P = setup(spec, pkl, beta)
    print("g(empty)/3W = %.6f" % (P["g0"] / (3 * P["W"])))
    keep = np.nonzero(P["keep"])[0]
    best = None
    seeds = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else list(keep[:8])
    for s in seeds:
        X, hist = greedy(P, tmax, [s])
        ok = [t for t, g in hist if g >= 0]
        tb = max(ok) if ok else 1
        print("seed %d: largest t with g >= 0: %d; g/3W at t = %s" % (s, tb, ", ".join("%d: %+.5f" % (t, g) for t, g in hist if t % 3 == 0 or t >= tb - 1)[:400]))
        e_int = int(P["A"][np.ix_(X[:tb], X[:tb])].sum() // 2)
        print("   X[:%d] = %s  (internal edges %d)" % (tb, ",".join(str(x) for x in X[:tb]), e_int))
        if best is None or tb > best[0]: best = (tb, X)
    pickle.dump(best, open("scan/pick_extra_last.pkl", "wb"))
