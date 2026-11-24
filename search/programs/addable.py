"""Which non-edges of an induced piece of Q(p,e0) can be added without creating a K4 (each one alone)?
usage: python3 exp/addable.py SPEC
A non-edge {u,v} is addable iff the common neighbourhood of u and v (inside the piece) is an independent set.
Reports the number of addable non-edges by number of common neighbours (= new triangles)."""
import sys
sys.path.insert(0, "exp")
import numpy as np, gsym

spec = sys.argv[1]
A, perm, removed, meta = gsym.from_spec(spec)
n = len(A); keep = np.ones(n, bool); keep[list(removed)] = False
idx = np.nonzero(keep)[0]; B = A[np.ix_(idx, idx)].astype(np.int64); m = len(idx)
C = B @ B                                   # common neighbours
# number of edges inside the common neighbourhood of u,v: sum_{x,y} B[u,x]B[v,x]B[x,y]B[u,y]B[v,y] / 2
E = np.zeros((m, m), dtype=np.int64)
for u in range(m):
    Nu = np.nonzero(B[u])[0]
    Bu = B[np.ix_(Nu, Nu)]                  # link of u
    # for v: common nbrs = Nu & N(v); edges inside = (1/2) * sum over x,y in Nu of B[v,x] Bu[x,y] B[v,y]
    X = B[:, Nu]                            # m x |Nu|
    E[u] = np.einsum('vx,xy,vy->v', X, Bu, X) // 2
non = (B == 0) & ~np.eye(m, dtype=bool)
add = non & (E == 0)
iu = np.triu_indices(m, 1)
print("%s: %d vertices, %d edges, %d non-edges, addable (alone) %d" % (spec, m, int(B.sum()) // 2, int(non[iu].sum()), int(add[iu].sum())))
vals, cnt = np.unique(C[iu][add[iu]], return_counts=True)
print("addable non-edges by number of common neighbours:", dict(zip(vals.tolist(), cnt.tolist())))
vals, cnt = np.unique(C[iu][non[iu]], return_counts=True)
print("all non-edges by number of common neighbours:", dict(zip(vals.tolist(), cnt.tolist())))
np.save("scan/addable_tmp.npy", np.array([idx[iu[0][add[iu]]], idx[iu[1][add[iu]]], C[iu][add[iu]]]))
