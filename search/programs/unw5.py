"""Unweighted lambda_min(N) (D = Deg, unit weights) for given sets of removed K'-cosets.  usage: python3 exp/unw5.py rm0 set1 set2 ...  (set = 0,3,7,12,18)"""
import numpy as np, sys, time
sys.path.insert(0, ".")
from symsub import SymSub, Sub, cyclic_subgroups
n = 941
S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
G = Sub(n, S, verbose=False)
def order(a):
    x = a; k = 1
    while x != 1: x = (x * a) % n; k += 1
    return k
gen = next(a for a in range(2, n) if order(a) == n - 1)
h = pow(gen, 20, n); Kp = [pow(h, t, n) for t in range(47)]
rm0 = int(sys.argv[1])
for arg in sys.argv[2:]:
    cs = [int(c) for c in arg.split(",")]
    R = set((pow(gen, c, n) * s) % n for c in cs for s in Kp) | ({0} if rm0 else set())
    t0 = time.time()
    Sx = SymSub(G, h, R, verbose=False)
    mu = np.ones(Sx.no)
    eigs = Sx.lam_all(mu, kk=1)
    lam = min(v[0][0] for v in eigs.values())
    print("cosets %-16s rm0 %d: n' %d edges %d tri %d  lambda_min(N) unweighted %.5f  [%.0fs]" % (arg, rm0, Sx.nv, Sx.ne, Sx.nt, lam, time.time() - t0)); sys.stdout.flush()
