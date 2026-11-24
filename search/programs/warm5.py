"""Restrict K'-symmetric weights of the 4-coset base to a 5-coset graph (optionally minus vertex 0), evaluate, save.
usage: python3 exp/warm5.py weights.pkl c rm0 out.pkl"""
import sys, pickle, numpy as np, io, contextlib
sys.path.insert(0, ".")
from warm752 import build, tmask
from opt2 import evaluate
n = 941
d = pickle.load(open(sys.argv[1], "rb")); mu_o = np.array(d["mu"], float); D_o = np.array(d["D"], float)
c = int(sys.argv[2]); rm0 = int(sys.argv[3])
with contextlib.redirect_stdout(io.StringIO()):
    G, So = build(n, 47, [0, 3, 7, 12], 0)
    _, Sn = build(n, 47, [0, 3, 7, 12, c], rm0)
to = tmask(G, So)
orb_glob = -np.ones(len(G.tri), dtype=np.int64); orb_glob[np.nonzero(to)[0]] = So.tri_orbit
tn = tmask(G, Sn)
gi_new = np.nonzero(tn)[0]
_, first = np.unique(Sn.tri_orbit, return_index=True)
mu_n = mu_o[orb_glob[gi_new[first]]]
D_n = np.zeros(Sn.nrep)
ekn = np.nonzero(Sn.edge_rep >= 0)[0]
D_n[Sn.edge_rep[ekn]] = D_o[So.edge_rep[ekn]]
fac = 3 * mu_n.sum() / D_n.sum(); D_n *= fac
lam, _ = evaluate(Sn, mu_n, D_n, 2)
print("fifth coset %2d rm0 %d: %d vertices, theta %.5f, lambda_min %.5f (margin %+.5f)" % (c, rm0, Sn.nv, fac, lam, lam + 2 / 3))
pickle.dump({"mu": mu_n, "D": D_n, "lam": lam, "oa": Sn.oa, "oi": Sn.oi}, open(sys.argv[4], "wb"))
