"""Method B near the full-coset end: X = {0} + (coset c minus Y).  Fixed weights; Lanczos value of N''.
usage: python3 exp/addback.py lifted.pkl"""
import sys, pickle, numpy as np, time
sys.path.insert(0, "exp")
from sublib import *
d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"]); D = np.array(d["D"]); cos = list(d["coset"])
h_ = h
orb = [cos[0]]
while len(orb) < 47: orb.append(orb[-1] * h_ % n)
assert sorted(orb) == sorted(cos)
def margin(Y):
    X = [0] + [v for v in cos if v not in Y]
    T = Target(X)
    D2 = D * (3 * (T.mf * mu).sum() / (T.nf * D).sum())
    N2, _ = sparse_N2(T, mu, D2)
    return lanczos_min(N2, T.ein, 1500) + 2 / 3, len(T.F)
for Y in ([], [orb[0]], [orb[0], orb[1]], [orb[0], orb[2]], [orb[0], orb[5]], [orb[0], orb[11]], [orb[0], orb[23]], [orb[0], orb[1], orb[2]], [orb[0], orb[7], orb[19]], [orb[0], orb[5], orb[11], orb[23]]):
    t0 = time.time(); m, nF = margin(Y)
    print("kept %d vertices of the coset (offsets %s): %d vertices, |F| = %d, margin %+.6f  [%.0fs]" % (len(Y), [orb.index(v) for v in Y], 705 + len(Y), nF, m, time.time() - t0)); sys.stdout.flush()
