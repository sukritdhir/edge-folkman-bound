"""First-order cost of removing single vertices (or whole cosets) from a K'-symmetric optimum.
For the optimal pair (mu, D) of the graph (cosets, rm0) and a vertex set X, the same pair restricted to G - X is still
feasible (method B), with budget ratio r(X) = sum_{E''} D / (3 sum_{T in G''} mu).  r(X) / r(empty) - 1 is an upper
bound for the increase of beta*, and margin(X) >= margin - (2/3) (r(X)/r(empty) - 1) roughly.
usage: python3 exp/loads.py weights.pkl"""
import numpy as np, sys, pickle
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup, survival
n = 941
d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float)
cosets = [int(c) for c in d["cosets"]]; rm0 = int(d["rm0"])
S_, G_, gen, h, R_, Sx = setup(n, 47, cosets, rm0)
Kp = [pow(h, t, n) for t in range(47)]
nf0, mf0 = survival(G_, Sx, [])
r0 = (nf0 * D).sum() / (3 * (mf0 * mu).sum())
print("cosets %s rm0 %d: beta %.8f (pkl), budget ratio of the stored pair %.8f, lambda %.7f" % (cosets, rm0, d.get("beta", np.nan), r0, d.get("lam", np.nan)))
rows = []
for c in range(20):
    if c in cosets: continue
    x = pow(gen, c, n)
    nf, mf = survival(G_, Sx, [x]); r1 = (nf * D).sum() / (3 * (mf * mu).sum())
    X = [(x * k) % n for k in Kp]
    nf, mf = survival(G_, Sx, X); rc = (nf * D).sum() / (3 * (mf * mu).sum())
    rows.append((r1 / r0 - 1, c, rc / r0 - 1))
    print("coset %2d (class %d): one vertex: ratio increase %.6f (margin cost about %.6f);  whole coset: %.5f (per vertex %.6f)"
          % (c, c % 5, r1 / r0 - 1, (2 / 3) * (r1 / r0 - 1), rc / r0 - 1, (rc / r0 - 1) / 47))
rows.sort()
print("cheapest single vertex: coset %d, ratio increase %.6f; cheapest whole coset: %d (%.5f)" % (rows[0][1], rows[0][0], min(rows, key=lambda t: t[2])[1], min(r[2] for r in rows)))
