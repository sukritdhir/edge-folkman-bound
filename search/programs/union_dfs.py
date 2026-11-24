"""Complete depth-first search over K4-free unions of orbits of a cyclic multiplier group H (with negation) on Z_n, for
the groups whose search was truncated in union_scan.py (node budget 4000) or that have more orbits than its limit.
For every MAXIMAL K4-free union of density >= MIN_DENS the unweighted normalised ratio is computed (exp/circ_search.py).
usage: python3 exp/union_dfs.py list.txt out.log [budget]        list.txt: lines "n hord norbits"
"""
import sys, time, os
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np
from circ_search import Circ
from union_scan import cyclic_subgroups_all, sym_orbits

MIN_DENS = float(os.environ.get("MIN_DENS", "0.14"))
REPORT = float(os.environ.get("REPORT", "1.09"))


def search(n, orbs, budget, out, tag):
    orbs = [o for o in orbs if not (n % 2 == 0 and n // 2 in o)]          # the class n/2 is not handled by the ratio code
    C = Circ(n, orbs); N = len(orbs)
    sizes = [len(o) for o in orbs]
    nodes = [0]; best = [None]; nmax = [0]

    def dfs(start, chosen, Sb, size):
        nodes[0] += 1
        if nodes[0] > budget: return
        cs = set(chosen)
        ext = [i for i in range(N) if i not in cs and not C.k4_at(Sb | C.obits[i], C.reps[i])]
        if size + sum(sizes[i] for i in ext if i >= start) < MIN_DENS * n and any(i >= start for i in ext): return      # cannot become dense enough
        if not ext:
            if size >= MIN_DENS * n and size >= 12:
                nmax[0] += 1
                r = C.ratio(chosen)
                if best[0] is None or r[0] < best[0][0][0]: best[0] = (r, list(chosen))
                if r[0] < REPORT:
                    line = "n %4d %s  S = union of orbits %s  d %3d  tri/edge %d..%d  ratio %.4f %s" % (n, tag, [C.reps[i] for i in chosen], r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
                    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
            return
        for i in ext:
            if i < start: continue
            dfs(i + 1, chosen + [i], Sb | C.obits[i], size + sizes[i])
    dfs(0, [], 0, 0)
    return nodes[0], nmax[0], best[0]


if __name__ == "__main__":
    lst = [tuple(int(x) for x in l.split()[:3]) for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    out = open(sys.argv[2], "a"); budget = int(sys.argv[3]) if len(sys.argv) > 3 else 300000
    done = set()
    if os.path.exists(sys.argv[2]):
        for l in open(sys.argv[2]):
            if l.startswith("# done"):
                p = l.split(); done.add((int(p[2]), int(p[3]), int(p[4])))
    sys.setrecursionlimit(10000)
    for (n, hord, norb) in lst:
        if (n, hord, norb) in done: continue
        t0 = time.time(); seen = set(); res = []
        for H in cyclic_subgroups_all(n):
            if len(H) != hord: continue
            orbs = sym_orbits(n, H)
            if len(orbs) != norb: continue
            key = frozenset(frozenset(o) for o in orbs)
            if key in seen: continue
            seen.add(key)
            res.append(search(n, orbs, budget, out, "|H| %3d orbits %2d" % (hord, norb)))
        nodes = sum(r[0] for r in res); trunc = any(r[0] > budget for r in res)
        bests = [r[2] for r in res if r[2] is not None]
        b = min(bests, key=lambda x: x[0][0]) if bests else None
        line = "# done %d %d %d  nodes %d%s  maximal %d  best %s  [%.0fs]" % (n, hord, norb, nodes, " TRUNCATED" if trunc else "", sum(r[1] for r in res),
                                                                              ("ratio %.4f d %d" % (b[0][0], b[0][1])) if b else "none", time.time() - t0)
        print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
