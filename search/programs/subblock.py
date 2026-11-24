"""Experiment: lambda_min of the principal submatrix N[E''] versus the full base matrix N (same symmetric weights).
usage: python3 exp/subblock.py weights.pkl steps"""
import numpy as np, sys, pickle, time, scipy.sparse as sp
sys.path.insert(0, ".")
from opt4 import setup, survival

d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float); X = list(d["X"])
steps = int(sys.argv[2])
S, G_, gen, h, R, Sx = setup(941, 47, [0, 3, 7, 12])
ek = np.nonzero(Sx.edge_rep >= 0)[0]
E = G_.E
De = np.zeros(E); De[ek] = D[Sx.edge_rep[ek]]
tri = Sx.tri_kept; wt = mu[Sx.tri_orbit]
nf, mf = survival(G_, Sx, X)
print("budget ratio on G'': sum D / 3W = %.8f" % ((nf * D).sum() / (3 * (mf * mu).sum())))
rows = np.concatenate([tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 1], tri[:, 2], tri[:, 2]])
cols = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 2], tri[:, 0], tri[:, 1]])
vals = np.concatenate([wt] * 6)
M = sp.csr_matrix((vals, (rows, cols)), shape=(E, E))
dm = np.zeros(E); dm[ek] = 1 / np.sqrt(De[ek])
N = sp.diags(dm) @ M @ sp.diags(dm)
bad = np.zeros(941, dtype=bool); bad[X] = True
e2 = ek[~bad[G_.ends[ek, 0]] & ~bad[G_.ends[ek, 1]]]
print("edges base %d, target %d (F = %d)" % (len(ek), len(e2), len(ek) - len(e2)))


def lanczos(A, steps, seed=0, report=(100, 200, 400, 800, 1600, 3200, 6400, 12800)):
    rng = np.random.default_rng(seed)
    n = A.shape[0]; v = rng.standard_normal(n); v /= np.linalg.norm(v)
    vo = np.zeros(n); beta = 0.0; al = []; be = []
    t0 = time.time()
    for k in range(1, steps + 1):
        w = A @ v - beta * vo
        a = float(w @ v); w -= a * v
        al.append(a); beta = float(np.linalg.norm(w)); be.append(beta)
        vo = v; v = w / beta
        if k in report or k == steps:
            from scipy.linalg import eigvalsh_tridiagonal
            ev = eigvalsh_tridiagonal(np.array(al), np.array(be[:-1]), select="i", select_range=(0, min(5, k - 1)))
            print("  steps %5d  smallest Ritz values %s  [%.0fs]" % (k, np.array2string(ev, precision=6), time.time() - t0)); sys.stdout.flush()


for name, idx in (("full base", ek), ("target E''", e2)):
    A = N[idx][:, idx].tocsr()
    print(name, A.shape, "nnz", A.nnz); sys.stdout.flush()
    lanczos(A, steps)
