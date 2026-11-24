"""Strongly regular graphs from the binary Golay code / Steiner system S(4,7,23): K4 test and unweighted ratio.
  G253  blocks of S(4,7,23), adjacent iff they meet in 1 point          SRG(253,112,36,60)
  G176  blocks missing a fixed point, adjacent iff they meet in 1 point  SRG(176,70,18,34)
  McL   the McLaughlin graph SRG(275,112,30,56) (22 points, 77 hexads, 176 heptads)
  G162  second subconstituent of McL                                     SRG(162,56,10,24) (locally the Gewirtz graph)
  G105/G120 etc are not built here.
usage: python3 exp/golay_graphs.py
"""
import sys, itertools
sys.path.insert(0, "exp")
import numpy as np
import gen


def golay_blocks():
    g = [0, 1, 5, 6, 7, 9, 11]                       # generator polynomial 1 + x + x^5 + x^6 + x^7 + x^9 + x^11
    G = np.zeros((12, 23), dtype=np.uint8)
    for i in range(12):
        for j in g: G[i, (i + j) % 23] = 1
    words = np.zeros((4096, 23), dtype=np.uint8)
    for m in range(4096):
        bits = [(m >> i) & 1 for i in range(12)]
        words[m] = (np.array(bits, dtype=np.uint8) @ G) % 2
    wt = words.sum(axis=1)
    hist = np.bincount(wt, minlength=24)
    assert hist[7] == 253 and hist[8] == 506 and hist[0] == 1 and hist[1:7].sum() == 0, hist
    return words[wt == 7].astype(bool)


def srg_params(A):
    n = len(A); d = A.sum(axis=1); assert (d == d[0]).all()
    C = A.astype(np.int64) @ A.astype(np.int64)
    lam = set(C[A].tolist()); mu = set(C[~A & ~np.eye(n, dtype=bool)].tolist())
    return n, int(d[0]), sorted(lam), sorted(mu)


def report(name, A):
    T = gen.triangles(A)
    k4 = gen.k4_free(A, T)
    par = srg_params(A)
    if not k4:
        print("%s: parameters %s  triangles %d  HAS K4" % (name, par, len(T))); return
    r = gen.unweighted_ratio(A, tol=1e-9, T=T)
    print("%s: parameters %s  triangles %d  K4-free  edges in triangles %d  tri/edge %d..%d  ratio %.4f %s" % (
        name, par, len(T), r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else ""))
    sys.stdout.flush()


if __name__ == "__main__":
    B = golay_blocks()                                # 253 x 23
    I = B.astype(np.int64) @ B.astype(np.int64).T     # intersection sizes
    print("intersection sizes:", np.unique(I, return_counts=True))
    A253 = (I == 1)
    report("G253 (meet in 1)", A253)
    A253b = (I == 3)
    report("G253 (meet in 3)", A253b)
    hept = np.nonzero(~B[:, 0])[0]; hexa = np.nonzero(B[:, 0])[0]
    assert len(hept) == 176 and len(hexa) == 77
    report("G176 (heptads meeting in 1)", A253[np.ix_(hept, hept)])
    report("G176 (heptads meeting in 3)", A253b[np.ix_(hept, hept)])
    report("G77 (hexads meeting in {inf} only)", A253[np.ix_(hexa, hexa)])
    report("G77 complement (hexads meeting in 3)", A253b[np.ix_(hexa, hexa)])
    # McLaughlin graph: points 1..22, hexads (blocks through 0), heptads (blocks missing 0)
    n = 22 + 77 + 176; A = np.zeros((n, n), dtype=bool)
    P = np.arange(22); HX = 22 + np.arange(77); HP = 99 + np.arange(176)
    for a, bl in enumerate(hexa):
        for p in range(22):
            if not B[bl, p + 1]: A[p, HX[a]] = A[HX[a], p] = True
    for a, bl in enumerate(hept):
        for p in range(22):
            if B[bl, p + 1]: A[p, HP[a]] = A[HP[a], p] = True
    A[np.ix_(HX, HX)] = (I[np.ix_(hexa, hexa)] == 1)
    A[np.ix_(HP, HP)] = (I[np.ix_(hept, hept)] == 1)
    X = (I[np.ix_(hexa, hept)] == 3)
    A[np.ix_(HX, HP)] = X; A[np.ix_(HP, HX)] = X.T
    assert (A == A.T).all() and not A.diagonal().any()
    report("McL", A)
    v = 0
    non = np.nonzero(~A[v])[0]; non = non[non != v]
    A162 = A[np.ix_(non, non)]
    report("G162 (second subconstituent of McL)", A162)
    nb = np.nonzero(A[v])[0]
    report("G112 (first subconstituent of McL)", A[np.ix_(nb, nb)])
    # complement-type relations of G162 and G176
    n2 = len(A162); report("G162 complement", ~A162 & ~np.eye(n2, dtype=bool))
