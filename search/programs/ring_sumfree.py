"""List the sum-free subgroups K (with -1) of small index in the unit groups of product rings.
usage: python3 exp/ring_sumfree.py lo hi out.log [maxindex] [kmin]"""
import sys, time
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np
from ring_dfs import Ring
import ringq
lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = open(sys.argv[3], "a")
maxidx = int(sys.argv[4]) if len(sys.argv) > 4 else 6; kmin = int(sys.argv[5]) if len(sys.argv) > 5 else 12
for sp in sorted(ringq.specs(lo, hi), key=lambda s: np.prod([c[1] for c in s])):
    t0 = time.time(); R = Ring(sp); n = R.n; U = len(R.units)
    psi = 1
    for kind, q in sp:
        p = ringq.prime_power(q); psi *= (q + 1) if kind == "F" else (q + q // p)
    found = []
    for K in R.subgroups():
        k = len(K)
        if k < kmin or U // k > maxidx: continue
        Kl = np.array(sorted(K)); inK = np.zeros(n, dtype=bool); inK[Kl] = True
        if inK[R.add[np.ix_(Kl, Kl)]].any(): continue
        # images in the components: index of the projection of K in each component's unit group
        found.append((U // k, k))
    for (e, k) in sorted(set(found)):
        line = "%s |R| %d units %d psi %d: sum-free K of index %d, |K| = %d, n = %d  (x%d)" % (R.name, n, U, psi, e, k, e * psi, found.count((e, k)))
        print(line); out.write(line + "\n"); out.flush()
    sys.stdout.flush()
print("done")
