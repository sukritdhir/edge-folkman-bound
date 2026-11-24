"""The U4(3) graph srg(162,56,10,24) (locally Gewirtz, hence K4-free) from PG(2,4): vertices = infinity, 56 hyperovals
of one class, 105 flags.  Checks the parameters and computes lambda_min of the edge-triangle matrix N = M / t."""
import numpy as np, itertools, sys
# GF(4) = {0,1,w,w^2}, encoded 0,1,2,3; addition = xor; multiplication table
mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
def add(a,b): return a ^ b
pts = []
for v in itertools.product(range(4), repeat=3):
    if v == (0,0,0): continue
    # normalise: first nonzero coordinate = 1
    f = next(x for x in v if x)
    inv = {1:1,2:3,3:2}[f]
    w = tuple(mul[inv][x] for x in v)
    if w == v: pts.append(v)
assert len(pts) == 21
def dot(a,b):
    s = 0
    for x,y in zip(a,b): s ^= mul[x][y]
    return s
lines = [frozenset(i for i,p in enumerate(pts) if dot(l,p) == 0) for l in pts]
assert all(len(L) == 5 for L in lines)
coll = set()
for L in lines:
    for tr in itertools.combinations(sorted(L), 3): coll.add(tr)
# hyperovals: 6 points, no three collinear
hyp = []
def ext(cur, start):
    if len(cur) == 6: hyp.append(frozenset(cur)); return
    for i in range(start, 21):
        if all((a,b,i) not in coll for a,b in itertools.combinations(cur, 2)): ext(cur + [i], i + 1)
ext([], 0)
print(len(hyp), "hyperovals")
# classes: even intersection
cls = [None] * len(hyp); k = 0
for i in range(len(hyp)):
    if cls[i] is None:
        cls[i] = k
        for j in range(len(hyp)):
            if cls[j] is None and len(hyp[i] & hyp[j]) % 2 == 0: cls[j] = k
        k += 1
print("classes:", [cls.count(c) for c in range(k)])
H = [hyp[i] for i in range(len(hyp)) if cls[i] == 0]
flags = [(p, li) for li, L in enumerate(lines) for p in L]
V = [("inf",)] + [("h", i) for i in range(len(H))] + [("f", i) for i in range(len(flags))]
n = len(V); A = np.zeros((n, n), dtype=np.int8)
for i in range(len(H)):
    A[0, 1 + i] = A[1 + i, 0] = 1
    for j in range(i):
        if not (H[i] & H[j]): A[1 + i, 1 + j] = A[1 + j, 1 + i] = 1
    for j, (p, li) in enumerate(flags):
        if p not in H[i] and len(lines[li] & H[i]) == 2: A[1 + i, 57 + j] = A[57 + j, 1 + i] = 1
for i, (p, li) in enumerate(flags):
    for j, (p2, lj) in enumerate(flags):
        if j >= i or p == p2 or li == lj: continue
        a = p2 in lines[li]; b = p in lines[lj]
        if a != b: A[57 + i, 57 + j] = A[57 + j, 57 + i] = 1
deg = A.sum(axis=1); A2 = A.astype(np.int32) @ A.astype(np.int32)
lam = set(A2[A == 1].tolist()); mu = set(A2[(A == 0) & ~np.eye(n, dtype=bool)].tolist())
print("n", n, "degrees", set(deg.tolist()), "lambda", lam, "mu", mu)
np.save("scan/u43_adj.npy", A)
