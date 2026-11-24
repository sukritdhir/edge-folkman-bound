"""Scan of the family Q(p; V0, S): vertices = nonzero vectors of GF(p)^2 modulo scalars from V0 (subgroup of index e0
of GF(p)^*, -1 in V0), [x] ~ [y] iff det(x, y) in S, S a union of cosets g^c V0 (given by the set of class numbers c).
The graph is vertex-transitive (SL(2,p)), so it is K4-free iff the link of one vertex is triangle-free.  S and c S
give isomorphic graphs, so class 0 is always in S; sets S with a non-trivial stabiliser are blow-ups and are skipped.
For every maximal... every K4-free union with at least MINTRI triangles on some edge the unweighted normalised ratio is
computed with the sparse eigensolver of exp/sl2_graph.py (below 1 = the graph arrows (3,3)).
usage: python3 exp/qscan.py plo phi out.log [nmax] [e0max]
"""
import sys, os, time
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np
import sl2_graph as sg
from qfam import prim_root

MINTRI = int(os.environ.get("MINTRI", "8"))
REPORT = float(os.environ.get("REPORT", "1.25"))


def class_matrix(p, e0):
    g = prim_root(p); V0 = sorted(set(pow(g, e0 * i, p) for i in range((p - 1) // e0)))
    cl = np.full(p, -1, dtype=np.int8); y = 1
    for i in range(p - 1): cl[y] = i % e0; y = (y * g) % p
    vid = {}; reps = []
    for a in range(p):
        for b in range(p):
            if (a, b) == (0, 0) or (a, b) in vid: continue
            for h in V0: vid[((a * h) % p, (b * h) % p)] = len(reps)
            reps.append((a, b))
    R = np.array(reps, dtype=np.int64)
    D = (R[:, 0][:, None] * R[:, 1][None, :] - R[:, 1][:, None] * R[:, 0][None, :]) % p
    return cl[D]                                              # -1 where det = 0


def scan(p, e0, out):
    n = (p + 1) * e0; CL = class_matrix(p, e0); assert CL.shape == (n, n)
    row0 = CL[0]; res = []; tested = [0]
    def k4free(S):
        N = np.nonzero(np.isin(row0, S))[0]
        L = np.isin(CL[np.ix_(N, N)], S)
        Li = L.astype(np.int32)
        return not ((Li @ Li) * Li).any()
    def canon(S):
        return min(tuple(sorted((c + s) % e0 for c in S)) for s in range(e0) if (-s) % e0 in S or True)
    seen = set()
    def dfs(S, start):
        for c in range(start, e0):
            S2 = S + [c]
            if not k4free(S2): continue
            key = min(tuple(sorted((x - s) % e0 for x in S2)) for s in S2)      # rotations that keep class 0 in S
            if key not in seen:
                seen.add(key)
                stab = [s for s in range(1, e0) if set((x + s) % e0 for x in S2) == set(S2)]
                if not stab: res.append(list(key))
            dfs(S2, c + 1)
    if k4free([0]):
        seen.add((0,)); res.append([0]); dfs([0], 1)
    best = None
    for S in sorted(res, key=lambda S: (-len(S), S)):
        A = np.isin(CL, S); d = int(A[0].sum())
        N = np.nonzero(A[0])[0]; kmax = int(A[np.ix_(N, N)].sum(axis=1).max())
        if kmax < MINTRI: continue
        t0 = time.time(); r = sg.ratio(A, k=4); tested[0] += 1
        if best is None or r[0] < best[0]: best = (r[0], S, d, r[2], r[3])
        if r[0] < REPORT:
            line = "Q(%d; index %d, classes %s)  n %d  d %d  tri/edge %d..%d  ratio %.5f %s [%.0fs]" % (p, e0, S, n, d, r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "", time.time() - t0)
            print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
    line = "# done p %d e0 %d n %d  K4-free unions (up to scaling, no blow-ups) %d  tested %d  best %s" % (p, e0, n, len(res), tested[0], ("ratio %.5f classes %s d %d tri/edge %d..%d" % best) if best else "none")
    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()


if __name__ == "__main__":
    plo, phi = int(sys.argv[1]), int(sys.argv[2]); out = open(sys.argv[3], "a")
    nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 1000; e0max = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    done = set((int(l.split()[3]), int(l.split()[5])) for l in open(sys.argv[3]) if l.startswith("# done"))
    for p in range(max(plo, 5), phi + 1):
        if any(p % r == 0 for r in range(2, int(p ** 0.5) + 1)): continue
        for e0 in range(2, min(e0max, p - 1) + 1):
            if (p - 1) % e0 or ((p - 1) // e0) % 2 or (p + 1) * e0 > nmax or (p, e0) in done: continue
            scan(p, e0, out)
