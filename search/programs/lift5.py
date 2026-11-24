"""Put K'-symmetric weights of the 705-vertex graph (cosets 0,3,7,12,c and vertex 0 removed) back on the 753-vertex base
(cosets 0,3,7,12): orbits inside the 705 graph take the new values, the other orbits keep (rescaled) old values.
usage: python3 exp/lift5.py weights705.pkl weights753_old.pkl out.pkl [c=18]"""
import sys, pickle, numpy as np, io, contextlib
sys.path.insert(0, ".")
from warm752 import build, tmask
n = 941
d5 = pickle.load(open(sys.argv[1], "rb")); d4 = pickle.load(open(sys.argv[2], "rb"))
c = int(sys.argv[4]) if len(sys.argv) > 4 else 18
with contextlib.redirect_stdout(io.StringIO()):
    G, So = build(n, 47, [0, 3, 7, 12], 0)
    _, Sn = build(n, 47, [0, 3, 7, 12, c], 1)
mu5 = np.array(d5["mu"], float); D5 = np.array(d5["D"], float)
mu4 = np.array(d4["mu"], float); D4 = np.array(d4["D"], float)
to, tn = tmask(G, So), tmask(G, Sn)
orb_glob = -np.ones(len(G.tri), dtype=np.int64); orb_glob[np.nonzero(to)[0]] = So.tri_orbit
gi_new = np.nonzero(tn)[0]
_, first = np.unique(Sn.tri_orbit, return_index=True)
idx_o = orb_glob[gi_new[first]]
cm = mu5.sum() / mu4[idx_o].sum()
mu = mu4 * cm; mu[idx_o] = mu5
ekn = np.nonzero(Sn.edge_rep >= 0)[0]
ea = So.edge_rep[ekn]; eb = Sn.edge_rep[ekn]
ua, fi = np.unique(ea, return_index=True)
cd = D5[eb[fi]].sum() / D4[ua].sum()
D = D4 * cd; D[ua] = D5[eb[fi]]
print("lifted: %d of %d triangle orbits and %d of %d edge orbits replaced; scale factors %.4f, %.4f" % (len(idx_o), So.no, len(ua), So.nrep, cm, cd))
kp = sorted(v for v in range(n) if not Sn.keep[v] and So.keep[v] and v != 0)
pickle.dump({"mu": mu, "D": D, "lam": None, "coset": kp}, open(sys.argv[3], "wb"))
