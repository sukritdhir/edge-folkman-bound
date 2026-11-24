"""Broader separation of triangle inequalities at a dual solution: for every representative edge e = (a, 0) take the
edges f with |Y_ef| > thr and test all pairs (f, g) among them with the four sign patterns.
usage: python3 exp/metric_sep2.py cosets rm0 y.npy [thr]"""
import numpy as np, sys, os, time, pickle
import scipy.sparse as sp
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup
n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
S_, G_, gen, h, R_, Sx = setup(n, 47, cosets, rm0)
q, nrep, no = Sx.q, Sx.nrep, Sx.no; NH = q // 2 + 1
oa = Sx.oa; oi = Sx.oi % q
y = np.load(sys.argv[3]); mu = y[:no]; dl = y[no:]
cols = np.repeat(np.arange(no), 3)
g = np.zeros((nrep, nrep, q), dtype=np.float32); dd = np.arange(q)
for r in range(NH):
    psi = np.exp(2j * np.pi * r * oi / q)
    U = sp.csc_matrix((np.conj(psi).ravel(), (oa.ravel(), cols)), shape=(nrep, no))
    S = (U @ sp.diags(mu) @ U.conj().T).toarray(); S[np.diag_indices(nrep)] += dl
    Gr = np.linalg.inv(S)
    ph = np.exp(-2j * np.pi * r * dd / q); cr = 1.0 if r == 0 else 2.0
    g += (cr / q) * (Gr.real[:, :, None] * ph.real[None, None, :] - Gr.imag[:, :, None] * ph.imag[None, None, :]).astype(np.float32)
dg = np.sqrt(g[np.arange(nrep), np.arange(nrep), 0].astype(float))
Y = g / (dg[:, None, None] * dg[None, :, None]).astype(np.float32)
np.save("scan/Y705.npy", Y)
tri = set()
for o in range(no):
    for p in range(3):
        for s in range(3):
            if s != p: tri.add((int(oa[o, p]), int(oa[o, s]), int((oi[o, s] - oi[o, p]) % q)))
t0 = time.time(); best = []; cnt = np.zeros(6, dtype=np.int64); edges_thr = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
for a in range(nrep):
    row = Y[a]                                                   # (nrep, q): Y between (a,0) and (b,d)
    bb, dv = np.nonzero(np.abs(row) > thr)
    keep = ~((bb == a) & (dv == 0)); bb = bb[keep]; dv = dv[keep]
    k = len(bb)
    if k < 2: continue
    ye = row[bb, dv].astype(np.float64)
    I, J = np.triu_indices(k, 1)
    yfg = Y[bb[I], bb[J], (dv[J] - dv[I]) % q].astype(np.float64)
    m4 = np.minimum(np.minimum(ye[I] + ye[J] + yfg, ye[I] - ye[J] - yfg), np.minimum(-ye[I] + ye[J] - yfg, -ye[I] - ye[J] + yfg))
    val = 3 + 2 * m4
    # exclude genuine triangles (three adjacent pairs with all-plus pattern giving exactly 1)
    for i, e in enumerate(edges_thr): cnt[i] += int((val < e - 1e-3).sum())
    idx = np.argsort(val)[:20]
    for ix in idx:
        if val[ix] < 0.9: best.append((float(val[ix]), a, int(bb[I[ix]]), int(dv[I[ix]]), int(bb[J[ix]]), int(dv[J[ix]]), float(ye[I[ix]]), float(ye[J[ix]]), float(yfg[ix])))
    if a % 200 == 0: print("  a = %d, list size %d, counts %s [%.0fs]" % (a, k, cnt.tolist(), time.time() - t0)); sys.stdout.flush()
best.sort()
print("counts of (e; f, g) with value below", edges_thr, ":", cnt.tolist(), "(each unordered triple is counted up to 3 times)")
for b in best[:25]:
    adj = [(b[1], b[2], b[3]) in tri, (b[1], b[4], b[5]) in tri, (b[2], b[4], (b[5] - b[3]) % q) in tri]
    print("  val %.4f  e=(%d,0) f=(%d,%d) g=(%d,%d)  Y_ef %.3f Y_eg %.3f Y_fg %.3f  adjacent pairs %s" % (b + (adj,)))
pickle.dump(best, open("scan/metric_viol2.pkl", "wb"))
