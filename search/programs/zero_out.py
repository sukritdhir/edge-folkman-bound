"""Start point for the add-back variant: K'-symmetric weights of the 705-vertex core (5-coset numbering) put on the
753-vertex base; the orbits outside the core get eps * old weights and D = 1.5 * (weighted degree inside the target),
so that the kept vertices are almost decoupled and budget-neutral (the value is then close to that of the core).
usage: python3 exp/zero_out.py weights705.pkl old753.pkl Xfile:m eps out.pkl"""
import sys, pickle, numpy as np, io, contextlib
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from warm752 import build, tmask
from sublib import *
d5 = pickle.load(open(sys.argv[1], "rb")); d4 = pickle.load(open(sys.argv[2], "rb"))
f_, m_ = sys.argv[3].split(":"); X = [int(v) for v in pickle.load(open(f_, "rb"))[int(m_)][0]]
eps = float(sys.argv[4])
with contextlib.redirect_stdout(io.StringIO()):
    G, So = build(n, 47, [0, 3, 7, 12], 0)
    _, Sn = build(n, 47, [0, 3, 7, 12, 18], 1)
mu5 = np.array(d5["mu"], float); D5 = np.array(d5["D"], float)
mu4 = np.array(d4["mu"], float)
to, tn = tmask(G, So), tmask(G, Sn)
orb_glob = -np.ones(len(G.tri), dtype=np.int64); orb_glob[np.nonzero(to)[0]] = So.tri_orbit
gi_new = np.nonzero(tn)[0]
_, first = np.unique(Sn.tri_orbit, return_index=True)
idx_o = orb_glob[gi_new[first]]
incore = np.zeros(So.no, dtype=bool); incore[idx_o] = True
mu = mu4 * (mu5.sum() / mu4[idx_o].sum()) * eps; mu[idx_o] = mu5
ekn = np.nonzero(Sn.edge_rep >= 0)[0]
ea = So.edge_rep[ekn]; eb = Sn.edge_rep[ekn]
ua, fi = np.unique(ea, return_index=True)
D = np.zeros(So.nrep); D[ua] = D5[eb[fi]]
# fringe orbits: 1.5 * weighted degree (triangles inside the target) of their alive members
T = Target(X)
wt = mu[T.torb_in]
deg = np.zeros(NE)
for j in range(3): np.add.at(deg, T.Lt_in[:, j], wt * (~incore[T.torb_in]))
degf = deg.reshape(nrep, q).max(axis=1)
out_orb = np.ones(nrep, dtype=bool); out_orb[ua] = False
D[out_orb] = np.maximum(1.5 * degf[out_orb], 1e-6 * D5.mean())
D2 = D * (3 * (T.mf * mu).sum() / (T.nf * D).sum())
N2, _ = sparse_N2(T, mu, D2)
lam = lanczos_min(N2, T.ein, 1500)
print("zero-out start (eps %.3f): %d vertices, budget factor %.6f, lambda_min(N'') = %.6f (margin %+.6f)" % (eps, Sx.nv - len(X), (3 * (T.mf * mu).sum() / (T.nf * D).sum()), lam, lam + 2 / 3))
pickle.dump({"mu": mu, "D": D2, "lam": lam, "X": X}, open(sys.argv[5], "wb"))
