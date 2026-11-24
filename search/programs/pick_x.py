"""Choose extra vertices X (a triangle, a diamond, ...) with the smallest fixed-weight budget ratio on the target,
for K'-constant weights (mu, D) of a base graph.   usage: python3 exp/pick_x.py cosets rm0 weights.pkl"""
import sys, pickle, numpy as np, itertools
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup
n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
S_, G_, gen, h, R_, Sx = setup(n, 47, cosets, rm0)
q, nrep, no = Sx.q, Sx.nrep, Sx.no
d0 = pickle.load(open(sys.argv[3], "rb")); mu = np.array(d0["mu"], float); D = np.array(d0["D"], float)
ek = np.nonzero(Sx.edge_rep >= 0)[0]; ends = G_.ends[ek]; De = D[Sx.edge_rep[ek]]
verts = np.nonzero(Sx.keep)[0]
adj = {int(v): set() for v in verts}; dsum = np.zeros(n); dedge = {}
for (u, v), d in zip(ends, De):
    u, v = int(u), int(v); adj[u].add(v); adj[v].add(u); dsum[u] += d; dsum[v] += d; dedge[(min(u, v), max(u, v))] = d
# triangles of the base as vertex triples, with their weights
te = Sx.tri_kept
tv = np.sort(np.concatenate([G_.ends[te[:, 0]], G_.ends[te[:, 1]], G_.ends[te[:, 2]]], axis=1), axis=1)
tri = np.stack([tv[:, 0], tv[:, 2], tv[:, 5]], axis=1)                 # sorted six entries a a b b c c
assert (tv[:, 0] == tv[:, 1]).all() and (tv[:, 2] == tv[:, 3]).all() and (tv[:, 4] == tv[:, 5]).all()
wt = mu[Sx.tri_orbit]
wsum = np.zeros(n); wedge = {}
for (a, b, c), w in zip(tri, wt):
    a, b, c = int(a), int(b), int(c)
    wsum[a] += w; wsum[b] += w; wsum[c] += w
    for p in ((a, b), (a, c), (b, c)):
        k = (min(p), max(p)); wedge[k] = wedge.get(k, 0.0) + w
wtri = {tuple(sorted((int(a), int(b), int(c)))): w for (a, b, c), w in zip(tri, wt)}
Dtot = De.sum(); Wtot = wt.sum()
print("base: beta = %.6f (sum D / 3 sum w)" % (Dtot / (3 * Wtot)))
def beta(X):
    X = list(X); Xs = set(X)
    dl = sum(dsum[x] for x in X) - sum(dedge[(min(a, b), max(a, b))] for a, b in itertools.combinations(X, 2) if b in adj[a])
    wl = sum(wsum[x] for x in X) - sum(wedge[(min(a, b), max(a, b))] for a, b in itertools.combinations(X, 2) if b in adj[a]) \
         + sum(wtri.get(tuple(sorted(t)), 0.0) for t in itertools.combinations(X, 3))
    return (Dtot - dl) / (3 * (Wtot - wl))
b1 = sorted((beta([v]), int(v)) for v in verts)
print("single vertices: best", [(round(b, 6), v) for b, v in b1[:5]], " worst %.6f" % b1[-1][0])
tl = sorted((beta(t), t) for t in wtri)
print("triangles: best", [(round(b, 6), t) for b, t in tl[:5]], " worst %.6f" % tl[-1][0])
# diamonds: triangle (a,b,c) plus a vertex d adjacent to two of its vertices; search around the best 3000 triangles
best4 = []
for b, t in tl[:3000]:
    for (x, y) in itertools.combinations(t, 2):
        for d in adj[x] & adj[y]:
            if d in t: continue
            best4.append((beta(list(t) + [d]), tuple(sorted(list(t) + [d]))))
best4 = sorted(set(best4))
print("diamonds: best", [(round(b, 6), t) for b, t in best4[:5]])
# K_{1,2,2}: diamond (a,b | c,d with ab the common edge?)  -> add a fifth vertex adjacent to as many as possible
best5 = []
for b, t in best4[:400]:
    cand = set()
    for x in t: cand |= adj[x]
    for e in cand - set(t):
        k = sum(1 for x in t if e in adj[x])
        if k >= 3: best5.append((beta(list(t) + [e]), tuple(sorted(list(t) + [e])), k))
best5 = sorted(set(best5))
print("5 vertices: best", [(round(b, 6), t, k) for b, t, k in best5[:5]])
best6 = []
for b, t, k in best5[:300]:
    cand = set()
    for x in t: cand |= adj[x]
    for e in cand - set(t):
        k2 = sum(1 for x in t if e in adj[x])
        if k2 >= 3: best6.append((beta(list(t) + [e]), tuple(sorted(list(t) + [e])), k2))
best6 = sorted(set(best6))
print("6 vertices: best", [(round(b, 6), t, k) for b, t, k in best6[:5]])
pickle.dump({"b1": b1[:50], "tri": tl[:200], "dia": best4[:200], "five": best5[:200], "six": best6[:200]}, open("scan/pick_x.pkl", "wb"))
