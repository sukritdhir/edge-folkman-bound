"""Induced subgraphs of Q(p, e0) on unions of 'split classes': the vertex [x], x = (a, b) in a fixed basis, has the
class ind(a b) mod 2 e0 (a b is defined up to squares of H); the vertices with a b = 0 (2 e0 'axis' vertices) form a
13th class 'A'.  The classes are the orbits of the split torus diag(t, 1/t).  Unweighted ratios of the induced
subgraphs for all class sets of the given sizes (up to J -> J + e0 and J -> -J... only J -> J + e0 is used).
usage: python3 exp/splitclass.py p e0 sizes out.log [report] [axis: 0/1]
"""
import sys, itertools, time
sys.path.insert(0, "exp")
import numpy as np
import qfam, gen


def split_classes(Q):
    p = Q.p; e0 = Q.e0; g = Q.g
    ind = np.full(p, -1, dtype=np.int64); y = 1
    for i in range(p - 1): ind[y] = i; y = (y * g) % p
    X = Q.X
    prod = (X[:, 0] * X[:, 1]) % p
    cl = np.where(prod == 0, -1, ind[prod] % (2 * e0))
    return cl


if __name__ == "__main__":
    p, e0 = int(sys.argv[1]), int(sys.argv[2])
    sizes = [int(x) for x in sys.argv[3].split(",")]; out = open(sys.argv[4], "a")
    report = float(sys.argv[5]) if len(sys.argv) > 5 else 1.2
    axis = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    Q = qfam.QGraph(p, e0); n = Q.n; A = Q.A; c = 2 * e0
    cl = split_classes(Q)
    cnt = np.bincount(cl[cl >= 0], minlength=c)
    line = "# Q(%d,%d) split classes: n = %d, class sizes %s, axis vertices %d" % (p, e0, n, cnt.tolist(), int((cl < 0).sum()))
    print(line); out.write(line + "\n"); out.flush()
    for s in sizes:
        seen = set(); best = None; num = 0; t0 = time.time()
        for J in itertools.combinations(range(c), s):
            key = min(tuple(sorted((j + k * e0) % c for j in J)) for k in range(c // e0))
            if key in seen: continue
            seen.add(key); num += 1
            sel = np.isin(cl, J)
            if axis: sel |= (cl < 0)
            idx = np.nonzero(sel)[0]
            r = gen.unweighted_ratio(A[np.ix_(idx, idx)], tol=1e-5)
            if best is None or r[0] < best[0]: best = (r[0], J, r, len(idx))
            if r[0] < report:
                line = "Q(%d,%d) split J %s axis %d  n %d  edges %d  tri/edge %d..%d  ratio %.4f %s" % (p, e0, list(J), axis, len(idx), r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
                print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
        line = "# done Q(%d,%d) split size %d axis %d: %d sets, best ratio %.4f J %s (n %d, tri/edge %d..%d)  [%.0fs]" % (p, e0, s, axis, num, best[0], list(best[1]), best[3], best[2][2], best[2][3], time.time() - t0)
        print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
