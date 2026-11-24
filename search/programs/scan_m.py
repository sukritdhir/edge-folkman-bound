"""Fixed symmetric weights; greedy X by budget cost; lambda_min of the sub-block N[E''] with the budget on G''.
usage: python3 exp/scan_m.py weights.pkl mmax [steps]"""
import numpy as np, sys, pickle, time, scipy.sparse as sp
from scipy.linalg import eigvalsh_tridiagonal
sys.path.insert(0, ".")
from opt4 import setup, survival

d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float); X = list(d.get("X", []))
mmax = int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
n = 941
S, G_, gen, h, R, Sx = setup(n, 47, [0, 3, 7, 12])
ek = np.nonzero(Sx.edge_rep >= 0)[0]; E = G_.E
De = np.zeros(E); De[ek] = D[Sx.edge_rep[ek]]
tri = Sx.tri_kept; wt = mu[Sx.tri_orbit]
tv = np.sort(np.concatenate([G_.ends[tri[:, 0]], G_.ends[tri[:, 1]], G_.ends[tri[:, 2]]], axis=1), axis=1)[:, [0, 2, 4]]
ebase = np.zeros(E, dtype=bool); ebase[ek] = True
rows = np.concatenate([tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 1], tri[:, 2], tri[:, 2]])
cols = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 2], tri[:, 0], tri[:, 1]])
M = sp.csr_matrix((np.concatenate([wt] * 6), (rows, cols)), shape=(E, E))
dm = np.zeros(E); dm[ek] = 1 / np.sqrt(De[ek])
N = (sp.diags(dm) @ M @ sp.diags(dm)).tocsr()


def stats(X):
    bad = np.zeros(n, dtype=bool); bad[list(X)] = True
    at = ~bad[tv].any(axis=1); ae = ebase & ~bad[G_.ends[:, 0]] & ~bad[G_.ends[:, 1]]
    return bad, at, ae


def next_vertex(X):
    bad, at, ae = stats(X)
    W = wt[at].sum(); Ds = De[ae].sum()
    wl = np.zeros(n); dl = np.zeros(n)
    for j in range(3): np.add.at(wl, tv[at, j], wt[at])
    aei = np.nonzero(ae)[0]
    for j in range(2): np.add.at(dl, G_.ends[aei, j], De[aei])
    th = 3 * (W - wl) / (Ds - dl); th[~Sx.keep] = -1; th[bad] = -1
    return int(np.argmax(th))


def lanczos_min(A, steps, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v)
    vo = np.zeros(A.shape[0]); beta = 0.0; al = []; be = []
    for k in range(steps):
        w = A @ v - beta * vo
        a = float(w @ v); w -= a * v
        al.append(a); beta = float(np.linalg.norm(w)); be.append(beta)
        vo = v; v = w / beta
    return eigvalsh_tridiagonal(np.array(al), np.array(be[:-1]), select="i", select_range=(0, 0))[0]


out = {}
while len(X) <= mmax:
    t0 = time.time()
    bad, at, ae = stats(X)
    fac = 3 * wt[at].sum() / De[ae].sum()
    idx = np.nonzero(ae)[0]
    lam = lanczos_min(N[idx][:, idx].tocsr(), steps) / fac
    out[len(X)] = (list(X), fac, lam)
    print("m %2d  vertices %d  edges %d  theta %.6f  lambda_sub %.6f  margin %+.6f  [%.0fs]"
          % (len(X), Sx.nv - len(X), len(idx), fac, lam, lam + 2 / 3, time.time() - t0)); sys.stdout.flush()
    pickle.dump(out, open("scan/subscan_fixed.pkl", "wb"))
    X = X + [next_vertex(X)]
