"""Separation of the 'diamond' inequalities at the dual solution Y (scan/Y705.npy from metric_sep2.py):
for two triangles T = {e,f,g}, T' = {e,f',g'} sharing the edge e and p = x_e+x_f+x_g, p' = x_e+x_f'+x_g' (both +-1 for
valid colourings):  (p + p' - x_e)^2 >= 1 etc. (triangle inequalities on the +-1 variables p, p', x_e)."""
import numpy as np, sys, pickle, time
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
S_, G_, gen, h, R_, Sx = setup(941, 47, cosets, rm0)
q, nrep, no = Sx.q, Sx.nrep, Sx.no
oa = Sx.oa; oi = Sx.oi % q
Y = np.load(sys.argv[3]).astype(np.float64)
tl = [[] for _ in range(nrep)]
for o in range(no):
    for p in range(3):
        s1, s2 = [s for s in range(3) if s != p]
        tl[oa[o, p]].append((oa[o, s1], (oi[o, s1] - oi[o, p]) % q, oa[o, s2], (oi[o, s2] - oi[o, p]) % q))
allv = []; cs = []; pps = []
for a in range(nrep):
    L = np.array(tl[a]); k = len(L)
    b1, d1, b2, d2 = L.T
    c = 1 + Y[a, b1, d1] + Y[a, b2, d2]                      # p . u_e
    cs.append(c)
    I, J = np.triu_indices(k, 1)
    cross = (Y[b1[I], b1[J], (d1[J] - d1[I]) % q] + Y[b1[I], b2[J], (d2[J] - d1[I]) % q]
             + Y[b2[I], b1[J], (d1[J] - d2[I]) % q] + Y[b2[I], b2[J], (d2[J] - d2[I]) % q])
    pp = c[I] + c[J] - 1 + cross
    pps.append(pp)
    v = np.stack([3 + 2 * (pp - c[I] - c[J]), 3 + 2 * (-pp + c[I] - c[J]), 3 + 2 * (-pp - c[I] + c[J]), 3 + 2 * (pp + c[I] + c[J])])
    allv.append(v.min(axis=0))
cs = np.concatenate(cs); pps = np.concatenate(pps); allv = np.concatenate(allv)
print("p.u_e over (edge, triangle) incidences: quantiles", np.round(np.quantile(cs, [0, .01, .1, .5, .9, .99, 1]), 3))
print("p.p' over diamonds (%d orbit representatives): quantiles" % len(pps), np.round(np.quantile(pps, [0, .001, .01, .1, .5, .9, .99, 1]), 3))
print("diamond inequality values (must be >= 1): quantiles", np.round(np.quantile(allv, [0, .001, .01, .1, .5]), 3),
      " fraction < 1: %.5f, < 0.9: %.5f, < 0.7: %.5f, < 0.5: %.6f" % ((allv < 1).mean(), (allv < .9).mean(), (allv < .7).mean(), (allv < .5).mean()))
