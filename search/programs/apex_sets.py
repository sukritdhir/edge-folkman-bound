"""Triangle-free unions of K'-cosets in G_941 = Cay(Z_941, 5th powers): candidate neighbourhoods for a NEW K'-fixed
vertex (an 'apex') that could be added to a K'-symmetric subgraph without creating a K4.
usage: python3 exp/apex_sets.py            (prints all MAXIMAL triangle-free unions, restricted to kept cosets or not)
"""
import numpy as np, itertools, sys
p = 941
H = sorted(set(pow(x, 5, p) for x in range(1, p)))
Kp = sorted(set(pow(302, i, p) for i in range(47))); assert len(Kp) == 47
cos = [sorted((pow(2, c, p) * k) % p for k in Kp) for c in range(20)]
assert sorted(x for c in cos for x in c) == list(range(1, p))
inH = np.zeros(p, dtype=bool); inH[H] = True
A = inH[(np.arange(p)[:, None] - np.arange(p)[None, :]) % p]
cosof = np.zeros(p, dtype=int)
for c in range(20):
    for x in cos[c]: cosof[x] = c
# edges between cosets: E[a][b] = number of edges between C_a and C_b (a != b) or inside (a == b)
E = np.zeros((20, 20), dtype=int)
for a in range(20):
    for b in range(20):
        E[a, b] = A[np.ix_(cos[a], cos[b])].sum()
print("edges between cosets: min %d max %d (off-diagonal), inside: %s" % (E[~np.eye(20, dtype=bool)].min(), E[~np.eye(20, dtype=bool)].max(), [int(E[a, a] // 2) for a in range(20)]))
A3 = None
def tri_free(J):
    V = [x for c in J for x in cos[c]]
    B = A[np.ix_(V, V)].astype(np.int64)
    return int(np.trace(B @ B @ B)) == 0
def edges(J):
    V = [x for c in J for x in cos[c]]
    return int(A[np.ix_(V, V)].sum() // 2)
res = []
def dfs(J, start):
    ext = [c for c in range(start, 20) if tri_free(J + [c])]
    anyext = [c for c in range(20) if c not in J and tri_free(J + [c])]
    if not anyext and J: res.append(list(J))
    for c in ext: dfs(J + [c], c + 1)
dfs([], 0)
print("maximal triangle-free unions of cosets: %d" % len(res))
kept = {0, 1, 3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19}
res.sort(key=lambda J: -edges(J))
for J in res[:60]:
    V = [x for c in J for x in cos[c]]
    B = A[np.ix_(V, V)]; deg = B.sum(axis=1)
    ev = np.linalg.eigvalsh(B.astype(float))
    print("J = %s  |V| = %d  edges %d  degree %d..%d  lambda_min %.3f  lambda_max %.3f  in kept: %s" % (J, len(V), edges(J), deg.min(), deg.max(), ev[0], ev[-1], set(J) <= kept))
