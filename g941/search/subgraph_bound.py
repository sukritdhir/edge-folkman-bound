"""Spectral arrowing test for induced subgraphs of a Lu-type circulant.

Criterion (valid for any graph and any real triangle weights mu_T with positive weighted degrees):
  M = sum_T mu_T * (adjacency of the triangle T on its three edges),  Deg_e = sum_{T containing e} mu_T,
  N = Deg^(-1/2) M Deg^(-1/2).   If lambda_min(N) > -2/3 then the graph arrows (3,3).
Proof: a colouring x in {+-1}^E without monochromatic triangle has x^T M x = -2 mu(T); with y = Deg^(1/2) x,
x^T M x = y^T N y >= lambda_min(N) * sum_e Deg_e = 3 lambda_min(N) mu(T).
"""
import numpy as np, scipy.sparse as sp, sys, time
from scipy.sparse.linalg import eigsh
from lu_scan import LuGraph, cyclic_subgroups


class Sub:
    def __init__(self, n, S, verbose=True):
        t0 = time.time()
        g = LuGraph(n, S); assert g.k4free(); g.setup()
        self.g = g; self.n = n; m = g.m; self.m = m
        Cp = np.array(g.Cp)
        # triangles through base edges (0,j): (j, j_f, x_f, j_g, x_g, type)
        rows = []
        for j, c in enumerate(g.Cp):
            for w in g.W:
                v = (c * w) % n
                out = [j]
                for (a, b) in ((0, v), (c, v)):
                    dd = (b - a) % n
                    if dd in g.cidx: out += [g.cidx[dd], a]
                    else: out += [g.cidx[(a - b) % n], b]
                out.append(g.typ[w]); rows.append(out)
        R = np.array(rows)
        x = np.arange(n)[:, None]
        e0 = (x * m + R[None, :, 0]).ravel()
        f = (((x + R[None, :, 2]) % n) * m + R[None, :, 1]).ravel()
        gg = (((x + R[None, :, 4]) % n) * m + R[None, :, 3]).ravel()
        ty = np.broadcast_to(R[None, :, 5], (n, R.shape[0])).ravel()
        T = np.sort(np.stack([e0, f, gg], axis=1), axis=1)
        key = (T[:, 0] * (n * m) + T[:, 1]) * (n * m) + T[:, 2]
        _, idx = np.unique(key, return_index=True)
        self.tri = T[idx]; self.tri_type = ty[idx]
        self.E = n * m
        self.ends = np.stack([np.repeat(np.arange(n), m), (np.repeat(np.arange(n), m) + np.tile(Cp, n)) % n], axis=1)
        if verbose:
            print("G: n %d d %d edges %d triangles %d (k = %d), triangle orbits %d  [%.1fs]" % (n, g.d, self.E, len(self.tri), g.k, g.T, time.time() - t0))

    def tri_vertices(self):
        a = self.ends[self.tri[:, 0]]; b = self.ends[self.tri[:, 1]]
        return np.sort(np.concatenate([a, b], axis=1), axis=1)       # 4 entries with one repeat; fine for masks

    def bound(self, keep_mask, mu=None, tol=1e-7, ncv=None):
        """keep_mask: boolean over vertices.  Returns (lambda_min(N), n', edges', triangles')."""
        ekeep = keep_mask[self.ends[:, 0]] & keep_mask[self.ends[:, 1]]
        tk = ekeep[self.tri[:, 0]] & ekeep[self.tri[:, 1]] & ekeep[self.tri[:, 2]]
        tri = self.tri[tk]
        w = np.ones(len(tri)) if mu is None else np.asarray(mu)[self.tri_type[tk]]
        E = self.E
        deg = np.zeros(E)
        for c in range(3): np.add.at(deg, tri[:, c], w)
        live = deg > 1e-12
        newid = -np.ones(E, dtype=np.int64); newid[live] = np.arange(int(live.sum()))
        r = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 0], tri[:, 2], tri[:, 1], tri[:, 2]])
        c = np.concatenate([tri[:, 1], tri[:, 0], tri[:, 2], tri[:, 0], tri[:, 2], tri[:, 1]])
        v = np.tile(w, 6)
        ne = int(live.sum())
        M = sp.coo_matrix((v, (newid[r], newid[c])), shape=(ne, ne)).tocsr()
        dm = 1.0 / np.sqrt(deg[live])
        Nn = sp.diags(dm) @ M @ sp.diags(dm)
        lam = eigsh(Nn, k=1, which="SA", tol=tol, ncv=ncv, return_eigenvectors=False)[0]
        return float(lam), int(keep_mask.sum()), int(ekeep.sum()), len(tri), ne


if __name__ == "__main__":
    n = 941
    S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
    G = Sub(n, S)
    full = np.ones(n, dtype=bool)
    t0 = time.time(); print("full graph: lambda_min(N) = %.5f (need > -0.66667)  [%.1fs]" % (G.bound(full)[0], time.time() - t0))
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    for r in (100, 155, 200, 250, 300):
        keep = full.copy(); keep[rng.choice(n, size=r, replace=False)] = False
        t0 = time.time(); lam, nv, ne, nt, live = G.bound(keep)
        print("random removal of %3d vertices: n' %d edges %d triangles %d  lambda_min(N) = %.5f  %s  [%.1fs]"
              % (r, nv, ne, nt, lam, "ARROWS" if lam > -2 / 3 else "-", time.time() - t0)); sys.stdout.flush()
