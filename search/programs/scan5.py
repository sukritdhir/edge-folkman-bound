"""Scan of sets of removed K'-cosets (vertex 0 removed too): unweighted lambda_min(N) (unit weights, D = Deg).
usage: python3 exp/scan5.py size part nparts out.log      (all canonical sets of `size` cosets, split into nparts)"""
import numpy as np, sys, time, itertools
import scipy.sparse as sp
from scipy.linalg import eigvalsh
from scipy.linalg.lapack import zpotrf
import os
THR = float(os.environ.get('THR', '-0.6920'))
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
size = int(sys.argv[1]); part = int(sys.argv[2]); nparts = int(sys.argv[3]); out = open(sys.argv[4], "a")
sets = []
for cs in itertools.combinations(range(20), size):
    canon = min(tuple(sorted((c + 5 * s) % 20 for c in cs)) for s in range(4))
    if canon == cs: sets.append(cs)
print(len(sets), "canonical sets"); sys.stdout.flush()
def proxy(cs):
    R = set((pow(gen, c, n) * s) % n for c in cs for s in Kp) | {0}
    Sx = SymSub(G, h, R, verbose=False)
    q, nrep, no = Sx.q, Sx.nrep, Sx.no
    oa = Sx.oa; oi = Sx.oi % q
    deg = np.bincount(oa.ravel(), minlength=nrep).astype(float)
    dm = 1 / np.sqrt(deg)
    cols = np.repeat(np.arange(no), 3)
    lam = 9.0; rmin = -1; Bs = []
    for r in range(q // 2 + 1):
        psi = np.exp(2j * np.pi * r * oi / q)
        U = sp.csc_matrix((np.conj(psi).ravel() * dm[oa.ravel()], (oa.ravel(), cols)), shape=(nrep, no))
        B = (U @ U.conj().T).toarray()                      # = Deg^(-1/2) (M + Deg) Deg^(-1/2) = N + I
        c, info = zpotrf(B - (1.0 + THR) * np.eye(nrep), lower=1)
        if info != 0: return None, r, Sx.ne, Sx.nt          # lambda_min(N) < THR in block r
        Bs.append(B)
    for r, B in enumerate(Bs):
        ev = eigvalsh(B, subset_by_index=[0, 0], check_finite=False)[0] - 1.0
        if ev < lam: lam = ev; rmin = r
    return lam, rmin, Sx.ne, Sx.nt
for i, cs in enumerate(sets):
    if i % nparts != part: continue
    t0 = time.time()
    lam, rmin, ne, nt = proxy(cs)
    out.write("%s %s r %d edges %d tri %d [%.1fs]\n" % (",".join(map(str, cs)), ("%.5f" % lam) if lam is not None else "fail", rmin, ne, nt, time.time() - t0)); out.flush()
