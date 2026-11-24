"""Expand a symmetric weight vector (one weight per triangle orbit) into an explicit certificate file (.npz):
n, S, removed, h, tri (vertex triples), w (weights)."""
import numpy as np, sys, pickle
from symsub import SymSub, Sub

def export(pkl, out):
    d = pickle.load(open(pkl, "rb"))
    n = d["n"]; S = d["S"]; h = d["h"]; R = d["removed"]; mu = np.array(d["mu"], dtype=float)
    G = Sub(n, S, verbose=False)
    Sx = SymSub(G, h, R, verbose=False)
    assert Sx.no == len(mu) and (Sx.oa == d["oa"]).all() and (Sx.oi == d["oi"]).all(), "orbit numbering mismatch"
    w = mu[Sx.tri_orbit]
    ends = G.ends[Sx.tri_kept]                               # (nt, 3, 2)
    verts = np.sort(ends.reshape(len(w), 6), axis=1)
    tri = verts[:, [0, 2, 4]]
    assert (verts[:, 0] == verts[:, 1]).all() and (verts[:, 2] == verts[:, 3]).all() and (verts[:, 4] == verts[:, 5]).all()
    scale = w.max(); w = w / scale
    extra = {}
    if "D" in d and d["D"] is not None:
        Dv = np.array(d["D"], dtype=float) / scale
        keptE = np.nonzero(Sx.edge_rep >= 0)[0]
        De = Dv[Sx.edge_rep[keptE]]
        live = De > 0
        De = De[live]; ed = np.sort(G.ends[keptE[live]], axis=1)
        De = De * min(1.0, 3 * w.sum() / De.sum())            # enforce sum D <= 3 w(T) exactly
        extra = {"edges": ed.astype(np.int32), "D": De}
    np.savez_compressed(out, n=n, S=np.array(S), removed=np.array(sorted(R)), h=h, tri=tri.astype(np.int32), w=w, **extra)
    print("wrote %s: %d vertices kept, %d triangles, lambda (from optimiser) %.5f" % (out, n - len(R), len(w), d["lam"]))

if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
