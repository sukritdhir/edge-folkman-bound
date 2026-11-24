"""Greedy choice of extra vertices X to remove from a K'-symmetric certificate (mu, D) on the base graph G':
minimise the budget loss  1 - theta(X),  theta(X) = 3 W''(X) / D''(X)   (W'' = triangle weight avoiding X,
D'' = diagonal on the edges avoiding X).  The pencil eigenvalue of the extension certificate is lambda / theta."""
import numpy as np, sys, pickle
from opt4 import setup
n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]
S, G, gen, h, R, Sx = setup(n, 47, cosets)
d = pickle.load(open(sys.argv[2], "rb")); mu = np.array(d["mu"], float)
D = np.array(d["D"], float) if "D" in d else Sx.degrees(mu)
D = D * (3 * mu.sum() / D.sum())
from opt2 import evaluate
lam = float(evaluate(Sx, mu, D, 1)[0])                   # pencil eigenvalue for the base-normalised diagonal
print("base-normalised lambda_min = %.6f" % lam); sys.stdout.flush()
mmax = int(sys.argv[3]); out = sys.argv[4]
ek = np.nonzero(Sx.edge_rep >= 0)[0]
De = np.zeros(G.E); De[ek] = D[Sx.edge_rep[ek]]
tri = Sx.tri_kept; wt = mu[Sx.tri_orbit]
tv = np.sort(np.concatenate([G.ends[tri[:, 0]], G.ends[tri[:, 1]], G.ends[tri[:, 2]]], axis=1), axis=1)
tv = np.stack([tv[:, 0], tv[:, 2], tv[:, 4]], axis=1)             # the three vertices of each triangle
assert (tv[:, 0] < tv[:, 1]).all() and (tv[:, 1] < tv[:, 2]).all()
alive_t = np.ones(len(tri), dtype=bool); alive_e = np.zeros(G.E, dtype=bool); alive_e[ek] = True
W = wt.sum(); Dsum = De[ek].sum()
X = []; removed = np.zeros(n, dtype=bool); res = {}
for step in range(mmax):
    wl = np.zeros(n); dl = np.zeros(n)
    for j in range(3): np.add.at(wl, tv[alive_t, j], wt[alive_t])
    ae = np.nonzero(alive_e)[0]
    for j in range(2): np.add.at(dl, G.ends[ae, j], De[ae])
    theta = 3 * (W - wl) / (Dsum - dl)                            # theta after removing each single vertex
    theta[~Sx.keep] = -1; theta[removed] = -1
    x = int(np.argmax(theta))
    X.append(x); removed[x] = True
    kill_t = alive_t & ((tv == x).any(axis=1)); W -= wt[kill_t].sum(); alive_t &= ~kill_t
    kill_e = alive_e & ((G.ends == x).any(axis=1)); Dsum -= De[kill_e].sum(); alive_e &= ~kill_e
    th = 3 * W / Dsum
    res[len(X)] = (list(X), th, lam / th)
    print("m %2d  add vertex %3d  theta %.6f  first-order lambda %.5f  margin %+.5f" % (len(X), x, th, lam / th, lam / th + 2 / 3)); sys.stdout.flush()
pickle.dump(res, open(out, "wb"))
