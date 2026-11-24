"""Diagnostic: how strongly does the dual solution of the K'-symmetric SDP violate the triangle (metric) inequalities
     (x_e + s x_f + s' x_g)^2 >= 1      for triples of edges that are not triangles of the graph?
Dual matrix Z ~ S^(-1) (from an interior-point iterate), normalised to unit diagonal: Y.  A triple with
v^T Y v = 3 + 2 (s Y_ef + s' Y_eg + s s' Y_fg) < 1 is a violated inequality that could be added to the certificate.
usage: python3 exp/metric_sep.py cosets rm0 y.npy
"""
import numpy as np, sys, os, time, pickle
import scipy.sparse as sp
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup

n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
S_, G_, gen, h, R_, Sx = setup(n, 47, cosets, rm0)
q, nrep, no = Sx.q, Sx.nrep, Sx.no; NH = q // 2 + 1
oa = Sx.oa; oi = Sx.oi % q
y = np.load(sys.argv[3]); mu = y[:no]; dl = y[no:]
cols = np.repeat(np.arange(no), 3)
t0 = time.time()
g = np.zeros((nrep, nrep, q), dtype=np.float32)
dd = np.arange(q)
for r in range(NH):
    psi = np.exp(2j * np.pi * r * oi / q)
    U = sp.csc_matrix((np.conj(psi).ravel(), (oa.ravel(), cols)), shape=(nrep, no))
    S = (U @ sp.diags(mu) @ U.conj().T).toarray(); S[np.diag_indices(nrep)] += dl
    Gr = np.linalg.inv(S)
    ph = np.exp(-2j * np.pi * r * dd / q)                               # g_ab(d) = (1/q) sum_r G_r[a,b] w^(-r d)
    cr = 1.0 if r == 0 else 2.0
    g += (cr / q) * (Gr.real[:, :, None] * ph.real[None, None, :] - Gr.imag[:, :, None] * ph.imag[None, None, :]).astype(np.float32)
dg = np.sqrt(g[np.arange(nrep), np.arange(nrep), 0].astype(float))
Y = g / (dg[:, None, None] * dg[None, :, None]).astype(np.float32)
print("Y built [%.0fs]; diag check %.6f" % (time.time() - t0, float(Y[5, 5, 0])))
# triangle sums
ts = np.zeros(no)
for (p, s) in [(0, 1), (0, 2), (1, 2)]:
    ts += Y[oa[:, p], oa[:, s], (oi[:, s] - oi[:, p]) % q]
print("triangle sums Y_ef + Y_eg + Y_fg over the triangle orbits: min %.4f mean %.4f max %.4f" % (ts.min(), ts.mean(), ts.max()))
pv = np.concatenate([Y[oa[:, p], oa[:, s], (oi[:, s] - oi[:, p]) % q] for (p, s) in [(0, 1), (0, 2), (1, 2)]])
print("Y on triangle pairs: quantiles", np.round(np.quantile(pv, [0, .01, .1, .5, .9, .99, 1]), 3))
# all entries
ab = np.abs(Y).ravel()
print("all |Y| entries: fraction > 0.1: %.5f, > 0.2: %.6f, > 0.3: %.6f, > 0.5: %.7f (diagonal %d of %d)" % ((ab > .1).mean(), (ab > .2).mean(), (ab > .3).mean(), (ab > .5).mean(), nrep, ab.size))
# neighbour lists: for the representative edge (a, 0): neighbours (b, d)
nb = [[] for _ in range(nrep)]
for o in range(no):
    for p in range(3):
        for s in range(3):
            if s != p: nb[oa[o, p]].append((oa[o, s], (oi[o, s] - oi[o, p]) % q))
nb = [sorted(set(l)) for l in nb]
print("neighbours per edge: mean %.1f" % np.mean([len(l) for l in nb]))
tri_pairs = set()
for a in range(nrep):
    for (b, d) in nb[a]: tri_pairs.add((a, b, d))
viol = []
t0 = time.time()
for a in range(nrep):
    l = nb[a]; k = len(l)
    bb = np.array([x[0] for x in l]); dv = np.array([x[1] for x in l])
    ye = Y[a, bb, dv].astype(float)                                     # Y_e,f for the neighbours
    I, J = np.triu_indices(k, 1)
    yfg = Y[bb[I], bb[J], (dv[J] - dv[I]) % q].astype(float)
    adj = np.array([(bb[i], bb[j], (dv[j] - dv[i]) % q) in tri_pairs for i, j in zip(I, J)])
    for (s1, s2) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        val = 3 + 2 * (s1 * ye[I] + s2 * ye[J] + s1 * s2 * yfg)
        # a triangle of the graph (all three pairs adjacent, signs ++): equality, skip
        ok = ~(adj & (s1 == 1) & (s2 == 1))
        for idx in np.nonzero(ok & (val < 0.9))[0]:
            viol.append((val[idx], a, int(bb[I[idx]]), int(dv[I[idx]]), int(bb[J[idx]]), int(dv[J[idx]]), s1, s2, bool(adj[idx])))
viol.sort()
vals = np.array([v[0] for v in viol]) if viol else np.zeros(0)
print("triples (e, f, g) with f, g neighbours of e [%.0fs]: violated (< 0.9): %d; < 0.7: %d; < 0.5: %d; < 0.3: %d; min %.4f"
      % (time.time() - t0, len(viol), int((vals < .7).sum()), int((vals < .5).sum()), int((vals < .3).sum()), vals.min() if len(vals) else 1))
for v in viol[:15]: print("   ", v)
pickle.dump(viol[:200000], open("scan/metric_viol.pkl", "wb"))
