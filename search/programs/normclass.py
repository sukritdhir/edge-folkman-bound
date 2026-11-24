"""Induced subgraphs of Q(p, e0) on unions of 'norm classes': the vertex i (= theta^i) has the class i mod c, where c
is a multiple of e0 dividing n = (p+1) e0 (c = 2 e0: the norm modulo squares of H; the classes are the orbits of the
shift by c).  For every set J of classes of the given sizes (up to the symmetry J -> J + e0) the unweighted normalised
ratio of the induced subgraph is computed.
usage: python3 exp/normclass.py p e0 c sizes out.log [report]       sizes like 6,7,8
"""
import sys, itertools, time
sys.path.insert(0, "exp")
import numpy as np
import qfam, gen

if __name__ == "__main__":
    p, e0, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sizes = [int(x) for x in sys.argv[4].split(",")]; out = open(sys.argv[5], "a")
    report = float(sys.argv[6]) if len(sys.argv) > 6 else 1.2
    Q = qfam.QGraph(p, e0); n = Q.n; A = Q.A
    assert n % c == 0 and c % e0 == 0
    cl = np.arange(n) % c
    line = "# Q(%d,%d): n = %d, %d classes of %d vertices (i mod %d)" % (p, e0, n, c, n // c, c)
    print(line); out.write(line + "\n"); out.flush()
    for s in sizes:
        seen = set(); best = None; cnt = 0; t0 = time.time()
        for J in itertools.combinations(range(c), s):
            key = min(tuple(sorted((j + k * e0) % c for j in J)) for k in range(c // e0))
            if key in seen: continue
            seen.add(key); cnt += 1
            idx = np.nonzero(np.isin(cl, J))[0]
            r = gen.unweighted_ratio(A[np.ix_(idx, idx)], tol=1e-5)
            if best is None or r[0] < best[0]: best = (r[0], J, r)
            if r[0] < report:
                line = "Q(%d,%d) c %d J %s  n %d  edges %d  tri/edge %d..%d  ratio %.4f %s" % (p, e0, c, list(J), len(idx), r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
                print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
        line = "# done Q(%d,%d) c %d size %d: %d sets, best ratio %.4f J %s (n %d, tri/edge %d..%d)  [%.0fs]" % (p, e0, c, s, cnt, best[0], list(best[1]), s * (n // c), best[2][2], best[2][3], time.time() - t0)
        print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
