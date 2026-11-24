"""Per-vertex first-order costs of removing a vertex at fixed symmetric weights:  c(v) = (3 w(v) - D(v)) / (3 W),
w(v) = weight of the triangles at v, D(v) = sum of D over the edges at v.  usage: python3 exp/vcost.py SPEC result.pkl"""
import sys, pickle
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np, gsym

def load(spec, pkl):
    A, perm, removed, meta = gsym.from_spec(spec)
    G_, Sx = gsym.make(A, perm, removed, drop_short=meta.get('drop_short', False))
    d = pickle.load(open(pkl, "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float)
    wt = mu[Sx.tri_orbit]; De = D[Sx.edge_rep]
    return A, perm, removed, G_, Sx, wt, De

if __name__ == "__main__":
    A, perm, removed, G_, Sx, wt, De = load(sys.argv[1], sys.argv[2])
    n = G_.n; W = wt.sum()
    wv = np.zeros(n); Dv = np.zeros(n)
    for j in range(3): np.add.at(wv, Sx.tri_verts[:, j], wt)
    for j in range(2): np.add.at(Dv, G_.ends[:, j], De)
    beta = De.sum() / (3 * W)
    cost = (3 * wv - Dv) / (3 * W)
    keep = np.nonzero(Sx.keep)[0]
    print("beta %.6f; vertices %d; mean cost %.6f (1/n = %.6f)" % (beta, len(keep), cost[keep].mean(), 1 / len(keep)))
    vals = np.round(cost[keep], 7); u, c = np.unique(vals, return_counts=True)
    print("distinct costs:", list(zip(u.tolist(), c.tolist()))[:20])
    q = Sx.q
    # classes of the vertices under the symmetry: orbit representatives
    seen = np.zeros(n, bool)
    for v in keep:
        if seen[v]: continue
        o = [v]; x = perm[v]
        while x != v: o.append(x); x = perm[x]
        seen[o] = True
        print("orbit of vertex %d (size %d): cost %.6f, degree %d, w(v)/W %.5f, D(v)/sumD %.5f" % (v, len(o), cost[v], int(A[v][Sx.keep].sum()), wv[v] / W, Dv[v] / De.sum()))
    # weights statistics
    print("triangle weights: min/mean %.3f, max/mean %.3f; share of weight in the top half of triangles %.3f" % (wt.min() / wt.mean(), wt.max() / wt.mean(), np.sort(wt)[len(wt) // 2:].sum() / W))
    k = np.bincount(Sx.tri_kept.ravel(), minlength=len(De))
    print("D versus triangles on the edge: corr %.3f; D/mean by k:" % np.corrcoef(De, k)[0, 1], [(int(kk), round(float(De[k == kk].mean() / De.mean()), 3)) for kk in np.unique(k)])
