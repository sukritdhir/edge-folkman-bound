"""Local search over vertex sets of Q(p, e0) that are unions of orbits of the split torus element of a given odd
order (default 23): minimise the unweighted ratio of the induced subgraph at a fixed number of orbits.
usage: python3 exp/orbsearch.py p e0 order start nswap out.log [seed] [maxsweeps]
   start: "J:2,3,4,5" (split classes), "O:0,5,7,..." (orbit numbers) or "R:24" (random set of 24 orbits)
   moves: all single swaps (one orbit out, one in); the best improving swap is taken (steepest descent);
   nswap = how many candidate swaps are evaluated per sweep (0 = all), chosen at random.
   With "D:k" appended to start (for example "J:2,3,4,5/D:1") k orbits are first dropped greedily (best single removal).
"""
import sys, time
sys.path.insert(0, "exp")
import numpy as np
import qfam, gen, gsym, splitclass


def setup(p, e0, order):
    Q = qfam.QGraph(p, e0)
    perm = gsym.split_perm(Q, order)
    orb = gsym.split_orbits(perm)
    size = np.bincount(orb)
    full = np.nonzero(size == order)[0]                 # the orbits of full length (the axis vertices are fixed or short)
    return Q, perm, orb, full


def ratio(Q, orb, sel_orbits):
    idx = np.nonzero(np.isin(orb, sel_orbits))[0]
    r = gen.unweighted_ratio(Q.A[np.ix_(idx, idx)], tol=1e-5)
    return r[0], len(idx), r[1], r[2], r[3]


if __name__ == "__main__":
    p, e0, order = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    start = sys.argv[4]; nswap = int(sys.argv[5]); out = open(sys.argv[6], "a")
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    maxsweeps = int(sys.argv[8]) if len(sys.argv) > 8 else 50
    rng = np.random.default_rng(seed)
    Q, perm, orb, full = setup(p, e0, order)
    cl = splitclass.split_classes(Q)
    def say(s):
        print(s); out.write(s + "\n"); out.flush(); sys.stdout.flush()
    parts = start.split("/")
    kind, arg = parts[0].split(":")
    if kind == "J":
        J = [int(x) for x in arg.split(",")]
        cur = sorted(set(int(o) for o in orb[np.isin(cl, J)]))
    elif kind == "O":
        cur = sorted(int(x) for x in arg.split(","))
    else:
        cur = sorted(int(x) for x in rng.choice(full, size=int(arg), replace=False))
    assert all(o in set(full.tolist()) for o in cur)
    val = ratio(Q, orb, cur)
    say("# Q(%d,%d) order %d: %d full orbits; start %s: %d orbits, n %d, edges %d, tri/edge %d..%d, ratio %.4f" % (
        p, e0, order, len(full), start, len(cur), val[1], val[2], val[3], val[4], val[0]))
    for extra in parts[1:]:
        kind2, arg2 = extra.split(":")
        if kind2 == "D":
            for _ in range(int(arg2)):
                best = None
                for o in cur:
                    v = ratio(Q, orb, [x for x in cur if x != o])
                    if best is None or v[0] < best[0][0]: best = (v, o)
                cur = [x for x in cur if x != best[1]]; val = best[0]
                say("drop orbit %d: %d orbits, n %d, edges %d, tri/edge %d..%d, ratio %.4f   O:%s" % (
                    best[1], len(cur), val[1], val[2], val[3], val[4], val[0], ",".join(map(str, cur))))
    t0 = time.time()
    tabu = set()
    for sweep in range(maxsweeps):
        outside = [int(o) for o in full if o not in set(cur)]
        if nswap < 0:
            # fast mode: rank the swaps by the sum of the single effects (one removal, one addition) and evaluate
            # only the best -nswap of them exactly
            rm = {a: ratio(Q, orb, [x for x in cur if x != a])[0] for a in cur}
            ad = {b: ratio(Q, orb, sorted(cur + [b]))[0] for b in outside}
            est = sorted((rm[a] + ad[b], a, b) for a in cur for b in outside if (a, b) not in tabu)
            cand = [(a, b) for (_, a, b) in est[:-nswap]]
        else:
            cand = [(a, b) for a in cur for b in outside]
            if nswap and nswap < len(cand):
                cand = [cand[i] for i in rng.choice(len(cand), size=nswap, replace=False)]
        best = None
        for (a, b) in cand:
            new = sorted([x for x in cur if x != a] + [b])
            v = ratio(Q, orb, new)
            if best is None or v[0] < best[0][0]: best = (v, new, a, b)
        if best[0][0] < val[0] - 1e-6:
            cur = best[1]; val = best[0]; tabu = set()
            say("sweep %d: swap %d -> %d: n %d, edges %d, tri/edge %d..%d, ratio %.4f   O:%s  [%.0fs]" % (
                sweep, best[2], best[3], val[1], val[2], val[3], val[4], val[0], ",".join(map(str, cur)), time.time() - t0))
        else:
            say("sweep %d: no improving swap among %d (best %.4f); local optimum ratio %.4f  O:%s  [%.0fs]" % (
                sweep, len(cand), best[0][0], val[0], ",".join(map(str, cur)), time.time() - t0))
            if nswap < 0:
                for (a, b) in cand: tabu.add((a, b))
                if len(tabu) >= min(len(cur) * len(outside), -8 * nswap): break
            elif not nswap or nswap >= len(cur) * len(outside): break
