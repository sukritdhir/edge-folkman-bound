"""Edge and triangle counts of all unions of k orbits (order `order` split element) of Q(p,e0), from the quotient
counts.  usage: python3 exp/orbdens.py p e0 order k [top]"""
import sys, itertools
sys.path.insert(0, "exp")
import numpy as np, qfam, gsym, splitclass

p, e0, order, k = (int(x) for x in sys.argv[1:5]); top = int(sys.argv[5]) if len(sys.argv) > 5 else 30
Q = qfam.QGraph(p, e0); A = Q.A.astype(np.float64); n = Q.n
perm = gsym.split_perm(Q, order); orb = gsym.split_orbits(perm)
size = np.bincount(orb); full = np.nonzero(size == order)[0]; r = len(full)
P = np.zeros((n, r)); 
for j, o in enumerate(full): P[orb == o, j] = 1
E = P.T @ A @ P                                    # E[i,j] = number of ordered adjacent pairs (u in i, v in j)
# T[i,j,k] = number of ordered triangles (u,v,w) in orbits i,j,k
AP = A @ P                                         # n x r: neighbours of u in orbit j
T = np.zeros((r, r, r))
for i in range(r):
    rows = np.nonzero(orb == full[i])[0]
    # for u in orbit i: sum over v~u in j, w~u in k, v~w  = (A[u] * P_j)^T A (A[u] * P_k)
    u = rows[0]                                    # all u in the orbit are equivalent
    Nu = np.nonzero(A[u])[0]
    L = A[np.ix_(Nu, Nu)]; Pn = P[Nu]
    T[i] = order * (Pn.T @ L @ Pn)
cl = splitclass.split_classes(Q)
ocl = [int(cl[np.nonzero(orb == o)[0][0]]) for o in full]
print("orbits:", [int(o) for o in full]); print("classes:", ocl)
res = []
for S in itertools.combinations(range(r), k):
    S = list(S)
    ed = E[np.ix_(S, S)].sum() / 2; tr = T[np.ix_(S, S, S)].sum() / 6
    res.append((3 * tr / ed, 2 * ed / (k * order), tuple(int(full[s]) for s in S)))
res.sort(reverse=True)
print("top by triangles per edge (tri/edge, average degree, orbits [classes]):")
seen = 0
for x in res[:top]:
    print("  %.3f  %.2f  %s  %s" % (x[0], x[1], ",".join(map(str, x[2])), sorted(ocl[list(full).index(o)] for o in x[2])))
np.save("scan/orbdens_%d_%d_%d_%d.npy" % (p, e0, order, k), np.array([(x[0], x[1]) + x[2] for x in res]))
