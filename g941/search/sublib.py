"""Library for method B (see exp/drive_sub.py for the description).  Base graph: G_941 minus the K'-cosets `COSETS`."""
import numpy as np, sys, time, pickle, os
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve, eigvalsh_tridiagonal
sys.path.insert(0, ".")
from opt4 import setup
from opt3 import simplex_qp

n = 941
COSETS = [int(c) for c in os.environ.get("COSETS", "0,3,7,12").split(",")]
GAP0 = 2.5e-4
S_, G_, gen, h, R_, Sx = setup(n, 47, COSETS)
q, nrep, no = Sx.q, Sx.nrep, Sx.no
NE = nrep * q
ek = np.nonzero(Sx.edge_rep >= 0)[0]
pos = -np.ones(G_.E, dtype=np.int64); pos[ek] = Sx.edge_rep[ek] * q + Sx.edge_off[ek]
Lt = pos[Sx.tri_kept]; torb = Sx.tri_orbit                         # triangles as layout indices
ends = np.zeros((NE, 2), dtype=np.int64); ends[pos[ek]] = G_.ends[ek]
tverts = np.sort(np.concatenate([ends[Lt[:, 0]], ends[Lt[:, 1]], ends[Lt[:, 2]]], axis=1), axis=1)[:, [0, 2, 4]]
NH = q // 2 + 1                                                    # characters 0..(q-1)/2 (the others are conjugates)


class Target:
    def __init__(self, X):
        bad = np.zeros(n, dtype=bool); bad[list(X)] = True
        self.ein = ~bad[ends[:, 0]] & ~bad[ends[:, 1]]
        self.tin = self.ein[Lt].all(axis=1)
        self.F = np.nonzero(~self.ein)[0]; self.aF = self.F // q; self.oF = self.F % q
        self.Lt_in = Lt[self.tin]; self.torb_in = torb[self.tin]
        self.agg = sp.csr_matrix((np.ones(len(self.torb_in)), (self.torb_in, np.arange(len(self.torb_in)))), shape=(no, len(self.torb_in)))
        alive = self.ein.reshape(nrep, q).sum(axis=1)
        self.nf = alive / q                                                 # surviving fraction of each edge orbit
        self.mf = np.bincount(self.torb_in, minlength=no) / q               # surviving fraction of each triangle orbit
        # core = edge orbits that lie completely in the target (K'-symmetric part); fringe = other target edges
        self.co = np.nonzero(alive == q)[0]; self.nco = len(self.co)
        self.core_idx = (self.co[:, None] * q + np.arange(q)[None, :]).ravel()
        part = np.zeros(nrep, dtype=bool); part[(alive > 0) & (alive < q)] = True
        self.fr = np.nonzero(self.ein & np.repeat(part, q))[0]; self.nfr = len(self.fr)


def next_vertex(mu, D, X):
    bad = np.zeros(n, dtype=bool); bad[list(X)] = True
    De = np.repeat(D, q); wt = mu[torb]
    at = ~bad[tverts].any(axis=1); ae = ~bad[ends[:, 0]] & ~bad[ends[:, 1]]
    W = wt[at].sum(); Ds = De[ae].sum()
    wl = np.zeros(n); dl = np.zeros(n)
    for j in range(3): np.add.at(wl, tverts[at, j], wt[at])
    aei = np.nonzero(ae)[0]
    for j in range(2): np.add.at(dl, ends[aei, j], De[aei])
    th = 3 * (W - wl) / (Ds - dl); th[~Sx.keep] = -1; th[bad] = -1
    return int(np.argmax(th))


def sparse_N2(T, mu, D):
    dme = np.repeat(1 / np.sqrt(D), q)
    w = mu[T.torb_in]; e, f, g = T.Lt_in.T
    rows = np.concatenate([e, e, f, f, g, g]); cols = np.concatenate([f, g, e, g, e, f])
    return sp.csr_matrix((np.tile(w, 6) * dme[rows] * dme[cols], (rows, cols)), shape=(NE, NE)), dme


def lanczos_min(A, mask, steps, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(NE) * mask; v /= np.linalg.norm(v)
    vo = np.zeros(NE); beta = 0.0; al = []; be = []
    for k in range(steps):
        w = A @ v - beta * vo
        a = float(w @ v); w -= a * v
        al.append(a); beta = float(np.linalg.norm(w)); be.append(beta)
        vo = v; v = w / beta
    return float(eigvalsh_tridiagonal(np.array(al), np.array(be[:-1]), select="i", select_range=(0, 0))[0])


class Solver:
    """(N'' - s)^(-1) for the principal submatrix N'' on the target edges.
    N'' - s = [[A, B], [B^T, C]] with A = core part (invariant under K', block-diagonal over its characters) and
    C = fringe part; solved through the Schur complement  Sf = C - B^T A^(-1) B  on the fringe."""
    def __init__(self, T, mu, D, s, N2, count=False):
        self.T = T; self.inv = []; self.nneg = 0
        big = 0.0
        for r in range(NH):
            Nr, idx, dm = Sx.block(mu, r, D)
            assert len(idx) == nrep
            A = Nr[np.ix_(T.co, T.co)] - s * np.eye(T.nco)
            if count:
                ev = np.linalg.eigvalsh(A); self.nneg += (1 if r == 0 else 2) * int((ev < 0).sum())
            Ai = np.linalg.inv(A); self.inv.append(Ai); big = max(big, float(np.abs(Ai).max()))
        self.big = big
        N2c = N2.tocsr()
        self.B = N2c[T.core_idx][:, T.fr].tocsc()                           # core x fringe (sparse)
        C = N2c[T.fr][:, T.fr].toarray() - s * np.eye(T.nfr)
        self.W = np.zeros((len(T.core_idx), T.nfr))
        for a in range(0, T.nfr, 256):
            self.W[:, a:a + 256] = self.core(self.B[:, a:a + 256].toarray())
        Sf = C - self.B.T @ self.W
        self.Sf = (Sf + Sf.T) / 2
        self.fac = lu_factor(self.Sf) if T.nfr else None

    def core(self, Y):
        k = Y.shape[1]
        Cf = np.fft.rfft(Y.reshape(self.T.nco, q, k), axis=1)
        for r in range(NH): Cf[:, r, :] = self.inv[r] @ Cf[:, r, :]
        return np.fft.irfft(Cf, n=q, axis=1).reshape(self.T.nco * q, k)

    def sub(self, Y):
        T = self.T
        zc = self.core(Y[T.core_idx])
        out = np.zeros_like(Y)
        if T.nfr:
            uf = lu_solve(self.fac, Y[T.fr] - self.B.T @ zc)
            out[T.fr] = uf
            zc -= self.W @ uf
        out[T.core_idx] = zc
        return out

    def count_sub(self):
        """number of eigenvalues of N'' below s  (needs count=True; Haynsworth: inertia(A) + inertia(Sf))"""
        return self.nneg + (int((np.linalg.eigvalsh(self.Sf) < 0).sum()) if self.T.nfr else 0)


def evaluate(T, mu, D, V, s, passes):
    """Returns (theta (k,), V (NE,k) Ritz vectors of N'', residual norms, rm (k,no), rd (k,nrep))."""
    t0 = time.time()
    N2, dme = sparse_N2(T, mu, D)
    sol = Solver(T, mu, D, s, N2)
    t1 = time.time()
    V = V * T.ein[:, None]
    for it in range(passes):
        V = sol.sub(V)
        V, _ = np.linalg.qr(V)
    NV = N2 @ V
    th, Q = np.linalg.eigh(V.T @ NV)
    V = V @ Q; NV = NV @ Q
    res = np.linalg.norm(NV - V * th[None, :], axis=0)
    t2 = time.time()
    rm, rd = cuts(T, V, dme)
    return th, V, res, rm, rd, (t1 - t0, t2 - t1, time.time() - t2, sol.big)


def cuts(T, V, dme):
    """Cut coefficients (per triangle orbit, per edge orbit) of the vectors x = D^(-1/2) u, u = columns of V (zero on F)."""
    Xu = V * dme[:, None]
    e, f, g = T.Lt_in.T
    rm = np.zeros((V.shape[1], no))
    for a in range(0, V.shape[1], 32):
        xs = Xu[:, a:a + 32]
        xe, xf, xg = xs[e], xs[f], xs[g]
        val = (xe + xf + xg) ** 2 - xe ** 2 - xf ** 2 - xg ** 2
        rm[a:a + 32] = (T.agg @ val).T
    rd = (2.0 / 3.0) * (Xu ** 2).reshape(nrep, q, -1).sum(axis=1).T
    return rm, rd


def certify(T, mu, D, tau_shift):
    """Inertia count: number of eigenvalues of N'' below -2/3 + tau_shift (0 means lambda_min(N'') > -2/3 + tau_shift)."""
    N2, _ = sparse_N2(T, mu, D)
    sol = Solver(T, mu, D, -2 / 3 + tau_shift, N2, count=True)
    return sol.count_sub(), sol.nneg
