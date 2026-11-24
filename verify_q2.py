#!/usr/bin/env python3
"""
verify_q2.py -- rigorous verifier for spectral certificates of upper bounds on the edge Folkman number
F_e(3,3;4) on the graphs Q(p; V0, S) and their induced subgraphs.  (Version 2 of verify_q.py.  Two changes: the
symmetry pi may also be given by a 2 x 2 matrix, and the weights w_T may be negative, because the proof of the Lemma
does not use their sign.  Everything else is unchanged, and certificates for verify_q.py are accepted.)

WHAT IS PROVED.  The certificate file describes a graph G'' and data (w, D).  If this program ends with
"THEOREM CHECKED", then G'' contains no K_4 and every colouring of the edges of G'' with two colours
contains a monochromatic triangle; hence F_e(3,3;4) <= |V(G'')|.

EVERY DECISIVE STEP USES EXACT INTEGER ARITHMETIC (Python integers and 64-bit integers with explicit
overflow guards).  Floating point is used only to *guess* auxiliary quantities (a shift and an approximate
Cholesky factor).  A wrong guess can make the program fail, never make it accept.

THE GRAPH.  p is an odd prime, nr a non-residue modulo p, and GF(p^2) = GF(p)[t]/(t^2 - nr); the element
a + b t is the vector (a, b).  theta = (a, b) is an element of GF(p^2), e0 divides p - 1, n = (p + 1) e0, and
    X_i = theta^i   (i = 0, ..., n-1),      V0 = {x in GF(p)^*: x^((p-1)/e0) = 1},      g = theta^(p+1) in GF(p)^*,
    S = union of the cosets g^c V0 over the listed classes c.
The graph Q has the vertices 0, ..., n-1, and i ~ j iff det(X_i, X_j) mod p lies in S (det((a,b),(c,d)) = ad - bc).
The program computes this adjacency matrix and checks that it is symmetric (the diagonal is zero because
det(X, X) = 0 is not in S).  Nothing else about the construction is used: the theorem is proved for the graph
with exactly this adjacency matrix.  (Background, not used: the classes V0 X_i are the nonzero vectors modulo
scalars from V0; this is checked as a consistency test only.)
    removed        vertices deleted from Q: this gives the BASE graph G'
    extra          (optional) further vertices X: the TARGET graph is G'' = G' - X
    step s         the permutation pi: i -> i + e0 s (mod n) of the vertices; the program checks that pi maps the
                   adjacency matrix and the set `removed` to themselves; q = order of pi
  or
    matrix a b c d the permutation pi of the vertices induced by the linear map (x0, x1) -> (a x0 + c x1, b x0 + d x1)
                   of GF(p)^2 (row vector times the matrix [[a, b], [c, d]]): pi(i) = j iff X_i times the matrix
                   equals v X_j for some v in V0.  The program checks that this defines a permutation of the
                   vertices and that it maps the adjacency matrix and the set `removed` to themselves (so nothing
                   about the matrix is assumed); q = order of pi.
    uniform        every triangle of G' has the weight 1, and D_e = number of triangles of G' on the edge e
  or
    a b c w        weight w (any rational number, of either sign) of the triangle {a,b,c} of G' and of its q images
                   under pi
    D u v d        (optional) a number d > 0 for the edge {u,v} and its q images.  Without D-lines,
                   D_e := sum of w_T over the listed triangles T containing e.
All decimal numbers in the file are read as exact rationals.

THE CRITERION (Lemma).  Let I be the set of edges e of G' that carry a number D_e > 0; every edge of a
listed triangle of nonzero weight must lie in I.  The weights w_T are arbitrary rational numbers.  For e != f in I put M[e,f] = w_T if e and f are two
edges of a listed triangle T (two edges lie in at most one triangle), and M[e,f] = 0 otherwise; M[e,e] = 0.
Let A = 3 M + 2 D  (D = diag(D_e); rows and columns indexed by I), and let P be the set of the edges in I
with both ends outside X (the indexed edges of G'').  If
    (i)  A is positive definite (then so is its principal submatrix A_PP), and
    (ii) sum of D_e over e in P   <=   3 * sum of w_T over the listed triangles T disjoint from X,
         and P is not empty,
then every 2-colouring of E(G'') has a monochromatic triangle.
Proof.  Let x in {+1,-1}^E(G'') be a colouring without monochromatic triangle, and extend it by 0 to the
other edges of G'.  A listed triangle T disjoint from X is a triangle of G'' (G'' is an induced subgraph);
its three edges do not all have the same colour, so the three pairwise products of their x-values sum
to -1 (two of the three edges have one colour and the third has the other: 1 - 1 - 1; this holds whatever the sign
of w_T is).  A listed triangle that meets X has at most one edge in G'', and all its pairwise products vanish.
Hence x^T M x = -2 W'', where W'' is the sum of w_T over the listed triangles T disjoint from X.  Also
x^T D x = sum of D_e over e in P, which is <= 3 W'' by (ii).  Therefore
x^T A x = 3 x^T M x + 2 x^T D x <= -6 W'' + 6 W'' = 0.  But x vanishes on the edges of I outside P and has no
zero entry on P (P is not empty), so x^T A x = x_P^T A_PP x_P > 0 by (i).  This is a contradiction.  QED
(In the uniform case with X empty: A = 3M + 2 Deg, where M is the 0/1 matrix "two edges lie in a common triangle"
and Deg_e is the number of triangles on e, and (ii) holds with equality.)

HOW (i) IS CHECKED RIGOROUSLY (the same method as METHOD A of verify_exact.py).
 (a) Symmetry reduction.  The map pi permutes I in orbits of full length q (checked) and A commutes with pi
     (checked: weights and D are constant on orbits by construction, and the triangle list is closed under pi).
     Choose representatives e_1..e_m of the orbits and put
         A_j[a,b] = A[e_a, pi^j e_b]      (integer matrices after clearing the common denominator),
         B_r = sum_j A_j * omega^(r j),  omega = exp(2 pi i / q),  r = 0..q-1   (Hermitian m x m).
     The vectors u_{r,a} = q^(-1/2) sum_j omega^(-r j) [pi^j e_a] form an orthonormal basis of C^I, and
     A u_{r,b} = sum_a conj(B_r[a,b]) u_{r,a}.  So A is unitarily similar to the direct sum of the matrices
     conj(B_r), and A is positive definite iff every B_r is.  Since B_{q-r} = conj(B_r), r <= q/2 suffices.
 (b) Real form.  Write B_r = X + iY (X symmetric, Y antisymmetric).  B_r is positive definite iff the real
     symmetric matrix R = [[X, -Y], [Y, X]] is; the two matrices have the same smallest eigenvalue.
 (c) Integer enclosure.  cos(2 pi k/q) and sin(2 pi k/q) are enclosed by exact rational arithmetic (Machin's
     formula for pi, Taylor series with Lagrange remainder), giving integers C_k, S_k with
     |C_k - 2^F cos| <= 1, |S_k - 2^F sin| <= 1.  With positive integers k_a (a diagonal scaling, any choice
     is valid) the integer matrix
         Rhat = round( 2^-Q [[Xn, -Yn],[Yn, Xn]] ),   Xn[a,b] = k_a k_b sum_j A_j[a,b] C_{rj},  Yn likewise with S,
     differs from the real matrix  Rreal = 2^(F-Q) K [[X,-Y],[Y,X]] K  (K = diag(k_a), twice)  by a symmetric
     matrix whose maximal absolute row sum is at most an explicitly computed integer `dlt`.
 (d) Exact residual.  From floating point we take an integer matrix Lhat and an integer sigma > 0 and compute,
     in exact integer arithmetic, E = Rhat - sigma*I - Lhat*Lhat^T.  For every unit vector v,
         v^T Rhat v = sigma + |Lhat^T v|^2 + v^T E v >= sigma - ||E||,   ||E|| <= max absolute row sum of E.
     Hence lambda_min(Rreal) >= sigma - ||E||_inf - dlt.  If this integer is positive, B_r is positive definite.

The routines for (c) and (d) (pi_enclosure, trig_tables, certify_pd) are those of verify_exact.py.

Usage:  python3 verify_q2.py certificate.txt [--blas-products] [--json out.json]
Needs Python >= 3.8 and numpy.  Never run with python3 -O (the checks are assert statements).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, json, hashlib, argparse, itertools, math
from fractions import Fraction
import numpy as np

assert __debug__, "do not run this program with python3 -O: the checks are assert statements"

F_BITS = 64          # trigonometric tables: integers C_k ~ 2^F cos(2 pi k / q)
G_BITS = 24          # scaling integers k_a ~ 2^G / sqrt(A[e_a, e_a])
UNIT_BITS = 60       # Rhat ~ 2^60 * (a matrix with diagonal about 1)
Q_BITS = 2 * G_BITS + F_BITS - UNIT_BITS
HALF = 30            # Lhat ~ 2^30 * (Cholesky factor), UNIT_BITS = 2 * HALF
LIMB = 15            # Lhat = H1 * 2^15 + L0


# ----------------------------------------------------------------------------------------------------
#  exact trigonometry
# ----------------------------------------------------------------------------------------------------
def _atan_inv_enclosure(x, nterms):
    """Enclosure of arctan(1/x), integer x >= 2: alternating series with decreasing terms."""
    s = Fraction(0)
    for i in range(nterms):
        s += Fraction((-1) ** i, (2 * i + 1) * x ** (2 * i + 1))
    nxt = Fraction((-1) ** nterms, (2 * nterms + 1) * x ** (2 * nterms + 1))   # first omitted term
    return (min(s, s + nxt), max(s, s + nxt))


def pi_enclosure(bits):
    """Rationals lo < pi < hi with hi - lo < 2^-bits (Machin: pi = 16 atan(1/5) - 4 atan(1/239))."""
    nterms = bits // 4 + 8
    a5 = _atan_inv_enclosure(5, nterms)
    a239 = _atan_inv_enclosure(239, nterms)
    lo = 16 * a5[0] - 4 * a239[1]
    hi = 16 * a5[1] - 4 * a239[0]
    assert Fraction(0) < hi - lo < Fraction(1, 2 ** bits)
    assert Fraction(314159265358979, 10 ** 14) < lo and hi < Fraction(314159265358980, 10 ** 14)
    return lo, hi


def _cos_sin_enclosure(theta_lo, theta_hi, bits):
    """Enclosures (lo, hi) of cos(theta) and of sin(theta) for some theta in [theta_lo, theta_hi] (0 <= theta < 8)."""
    assert 0 <= theta_lo <= theta_hi < 8
    two = 2 ** (bits + 8)
    t = Fraction(math.floor(theta_lo * two), two)                    # a dyadic rational, 0 <= theta - t <= w
    w = (theta_hi - theta_lo) + Fraction(1, two)
    tol = Fraction(1, 2 ** (bits + 8))
    # Taylor polynomials at t; the Lagrange remainder after the terms up to degree d is at most t^(d+1)/(d+1)!
    c = Fraction(0); s = Fraction(0); term = Fraction(1); d = 0     # term = t^d / d!
    while True:
        if d % 2 == 0: c += term if (d // 2) % 2 == 0 else -term
        else:          s += term if (d // 2) % 2 == 0 else -term
        d += 1
        term = term * t / d
        if d > 16 and d % 2 == 0 and term < tol and term * t / (d + 1) < tol:
            break
    rem_c = term                         # c has the terms of degree <= d-2 (and d-1 is odd): remainder <= t^d/d!
    rem_s = term                         # s has the terms of degree <= d-1: remainder <= t^d/d!
    # |cos(theta) - cos(t)| <= |theta - t| <= w, same for sin
    return (c - rem_c - w, c + rem_c + w), (s - rem_s - w, s + rem_s + w)


def trig_tables(q, F=F_BITS):
    """Integers C[k], S[k], k = 0..q-1, with |C[k] - 2^F cos(2 pi k/q)| <= 1 and |S[k] - 2^F sin(2 pi k/q)| <= 1,
    C[q-k] = C[k], S[q-k] = -S[k] exactly, C[0] = 2^F, S[0] = 0."""
    bits = F + 40
    plo, phi = pi_enclosure(bits + 8)
    C = [0] * q; S = [0] * q
    C[0] = 2 ** F; S[0] = 0
    for k in range(1, q // 2 + 1):
        if 2 * k == q:
            C[k] = -2 ** F; S[k] = 0
            continue
        (clo, chi), (slo, shi) = _cos_sin_enclosure(2 * k * plo / q, 2 * k * phi / q, bits)
        for (lo, hi, tab, sign) in ((clo, chi, C, 1), (slo, shi, S, -1)):
            assert hi - lo < Fraction(1, 2 ** (F + 2))               # half-width < 2^-(F+3)
            mid = (lo + hi) / 2
            val = math.floor(mid * 2 ** F + Fraction(1, 2))          # |val - 2^F mid| <= 1/2
            # |val - 2^F x| <= 1/2 + 2^F (hi - lo)/2 < 1 for every x in [lo, hi]
            tab[k] = val
            tab[q - k] = sign * val
        assert abs(C[k] / 2 ** F - math.cos(2 * math.pi * k / q)) < 1e-12      # sanity check only
        assert abs(S[k] / 2 ** F - math.sin(2 * math.pi * k / q)) < 1e-12
    return C, S


# ----------------------------------------------------------------------------------------------------
#  exact positive-definiteness check of an integer matrix
# ----------------------------------------------------------------------------------------------------
def matprod(A, B, blas):
    """Exact product of two int64 matrices whose entries and whose row-by-row sums of absolute products are
    below 2^50 (asserted by the caller through entry bounds).  Default: numpy's integer matrix product.
    With blas=True the product is computed by the BLAS in double precision; it is exact as well, because
    all entries are integers and every partial sum of products is an integer of absolute value < 2^53, so
    no floating-point operation rounds.  The default avoids even this argument."""
    if not blas:
        return A @ B
    P = A.astype(np.float64) @ B.astype(np.float64)
    assert np.abs(P).max() < 2.0 ** 52
    return P.astype(np.int64)


def certify_pd(Rhat, blas=False):
    """Rhat: symmetric int64 matrix, |entries| < 2^61, 'unit' 2^60.
    Returns (lb, info): lb is an integer with lambda_min(Rhat) >= lb, proved in exact integer arithmetic
    (lb is None if no proof was obtained)."""
    N = Rhat.shape[0]
    assert Rhat.dtype == np.int64 and Rhat.shape == (N, N) and N <= 4096
    assert bool((Rhat == Rhat.T).all()), "Rhat is not symmetric"
    assert int(np.abs(Rhat).max()) < 2 ** 61
    # ---------------- untrusted floating-point guess of (sigma, Lhat) ----------------
    Rf = Rhat.astype(np.float64) / 2.0 ** UNIT_BITS
    lam = float(np.linalg.eigvalsh(Rf)[0])
    info = {"float_lambda_min": lam}
    if not lam > 0:
        return None, info
    L = None
    for frac in (0.9, 0.75, 0.5):
        s = lam * frac
        try:
            L = np.linalg.cholesky(Rf - s * np.eye(N))
            break
        except np.linalg.LinAlgError:
            L = None
    if L is None:
        return None, info
    Lhat = np.rint(L * 2.0 ** HALF).astype(np.int64)
    sigma = int(s * 2.0 ** UNIT_BITS)
    del Rf, L
    if sigma <= 0:
        return None, info
    # ---------------- exact integer arithmetic from here on ----------------
    assert 0 < sigma < 2 ** 61
    base = 1 << LIMB
    L0 = np.mod(Lhat + base // 2, base) - base // 2            # in [-2^14, 2^14)
    H1 = (Lhat - L0) // base
    assert bool((H1 * base + L0 == Lhat).all())
    assert int(np.abs(H1).max()) < 2 ** 17 and int(np.abs(L0).max()) <= 2 ** 14
    # products: |entries| <= 4096 * 2^17 * 2^17 = 2^46: no overflow, and exact
    P11 = matprod(H1, H1.T, blas)
    P10 = matprod(H1, L0.T, blas)
    P00 = matprod(L0, L0.T, blas)
    P10s = P10 + P10.T
    assert int(np.abs(P11).max()) < 2 ** 31 and int(np.abs(P10s).max()) < 2 ** 45 and int(np.abs(P00).max()) < 2 ** 41
    Z = P11 * (1 << (2 * LIMB)) + P10s * (1 << LIMB) + P00    # = Lhat Lhat^T, |Z| < 2^61 + 2^60 + 2^41 < 2^62
    E = Rhat - Z                                               # |E| < 2^61 + 2^62 < 2^63
    idx = np.arange(N)
    E[idx, idx] -= sigma                                       # |.| < 2^61 + 2^62 + 2^61 = 2^63: guard below
    assert int(np.abs(Rhat[idx, idx]).max()) < 2 ** 61 and int(np.abs(Z[idx, idx]).max()) < 2 ** 61
    assert int(np.abs(E).max()) < 2 ** 50
    normE = int(np.abs(E).sum(axis=1).max())                   # < 4096 * 2^50 = 2^62
    info.update({"sigma": sigma, "normE": normE})
    return sigma - normE, info


def self_test():
    """K_6 arrows (3,3) and the criterion certifies it with unit weights; K_5 does not arrow, and the matrix
    3M + 2D is singular for unit weights, so no certificate may be accepted."""
    out = []
    for nv in (6, 5):
        edges = list(itertools.combinations(range(nv), 2)); eid = {e: i for i, e in enumerate(edges)}
        ne = len(edges)
        A = np.zeros((ne, ne), dtype=object)
        for T in itertools.combinations(range(nv), 3):
            es = [eid[p] for p in itertools.combinations(T, 2)]
            for e in es:
                A[e, e] += 2                        # 2 * D_e with D_e = number of triangles through e
                for f in es:
                    if e != f: A[e, f] += 3
        diag = int(A[0, 0])
        Rhat = np.array([[(int(A[i, j]) << UNIT_BITS) // diag for j in range(ne)] for i in range(ne)], dtype=np.int64)
        Rhat = (Rhat + Rhat.T) // 2
        lb, info = certify_pd(Rhat)
        out.append(lb)
    assert out[0] is not None and out[0] > 2 ** 40, out
    assert out[1] is None or out[1] <= 15, out
    return out


# ----------------------------------------------------------------------------------------------------
#  certificate and graph
# ----------------------------------------------------------------------------------------------------
def parse_certificate(fn):
    C = {"tri": [], "D": [], "uniform": False}
    with open(fn) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            key, *rest = line.split()
            if key in ("p", "nr", "e0", "step", "orbits", "edge_orbits"): C[key] = int(rest[0])
            elif key == "matrix":
                assert len(rest) == 4 and "matrix" not in C
                C["matrix"] = tuple(int(x) for x in rest)
            elif key == "family": assert rest == ["Q"], "unknown family"
            elif key == "theta": C["theta"] = (int(rest[0]), int(rest[1]))
            elif key in ("classes", "removed", "extra"): C[key] = C.get(key, []) + [int(x) for x in rest]
            elif key == "uniform": C["uniform"] = True
            elif key == "D": C["D"].append((int(rest[0]), int(rest[1]), Fraction(rest[2])))
            else: C["tri"].append((int(key), int(rest[0]), int(rest[1]), Fraction(rest[2])))
    for key in ("p", "nr", "theta", "e0", "classes"):
        assert key in C, "certificate lacks the line '%s'" % key
    assert ("step" in C) != ("matrix" in C), "the certificate needs exactly one of the lines 'step' and 'matrix'"
    C.setdefault("removed", []); C.setdefault("extra", [])
    assert not (C["uniform"] and (C["tri"] or C["D"])), "'uniform' excludes weight lines"
    if "orbits" in C and not C["uniform"]: assert len(C["tri"]) == C["orbits"]
    if "edge_orbits" in C and C["D"]: assert len(C["D"]) == C["edge_orbits"]
    return C


def bits_above(x, lo):
    """Positions > lo of the 1-bits of the nonnegative integer x, in increasing order."""
    x >>= lo + 1
    pos = lo + 1
    while x:
        low = x & -x
        b = low.bit_length() - 1
        yield pos + b
        x >>= b + 1
        pos += b + 1


def build_graph(C, log):
    p = C["p"]; nr = C["nr"] % p; e0 = C["e0"]; th = (C["theta"][0] % p, C["theta"][1] % p)
    assert p > 2 and all(p % r for r in range(2, int(p ** 0.5) + 1)), "p must be an odd prime"
    assert (p - 1) % e0 == 0 and e0 >= 1
    assert pow(nr, (p - 1) // 2, p) == p - 1, "nr must be a non-residue modulo p"
    n = (p + 1) * e0
    def mul(x, y): return ((x[0] * y[0] + nr * x[1] * y[1]) % p, (x[0] * y[1] + x[1] * y[0]) % p)
    X = []; cur = (1, 0)
    for i in range(n): X.append(cur); cur = mul(cur, th)
    g = (1, 0)
    for i in range(p + 1): g = mul(g, th)
    assert g[1] == 0 and g[0] != 0, "theta^(p+1) must lie in GF(p)^*"
    g = g[0]
    f = (p - 1) // e0
    V0 = [x for x in range(1, p) if pow(x, f, p) == 1]; assert len(V0) == f
    classes = sorted(set(c % e0 for c in C["classes"])); assert len(classes) == len(C["classes"])
    inS = np.zeros(p, dtype=bool)
    for c in classes:
        gc = pow(g, c, p)
        for v in V0: inS[(gc * v) % p] = True
    assert not inS[0]
    Xa = np.array(X, dtype=np.int64)
    Dm = (Xa[:, 0][:, None] * Xa[:, 1][None, :] - Xa[:, 1][:, None] * Xa[:, 0][None, :]) % p     # |products| < p^2 < 2^62
    assert p < 2 ** 30
    A = inS[Dm]
    assert bool((A == A.T).all()), "the adjacency matrix is not symmetric (S must equal -S)"
    assert not bool(A[np.arange(n), np.arange(n)].any())
    # consistency test only: the classes V0 X_i are pairwise disjoint
    seen = {}                                       # vector (as a * p + b) -> number of its class
    for i, (a, b) in enumerate(X):
        assert (a, b) != (0, 0)
        for v in V0:
            key = ((a * v) % p) * p + (b * v) % p
            assert key not in seen, "two of the vectors theta^i are equivalent modulo V0"
            seen[key] = i
    complete = (len(seen) == p * p - 1)
    removed = set(C["removed"]); extra = set(C["extra"])
    assert len(removed) == len(C["removed"]) and len(extra) == len(C["extra"])
    assert all(0 <= v < n for v in removed | extra) and not (removed & extra)
    V = [v for v in range(n) if v not in removed]; Vs = set(V)
    keepmask = sum(1 << v for v in V)
    adj = {}
    for v in V:
        row = int.from_bytes(np.packbits(A[v], bitorder="little").tobytes(), "little")
        adj[v] = row & keepmask
    nedges = 0; ntri = 0
    for u in V:
        for v in bits_above(adj[u], u):
            nedges += 1
            common = adj[u] & adj[v]
            for x in bits_above(common, v):
                ntri += 1
                assert (common & adj[x]) >> (x + 1) == 0, "the base graph contains a K4 on %d,%d,%d" % (u, v, x)
    degs = sorted(set(bin(adj[v]).count("1") for v in V))
    log("1. graph Q: p = %d, GF(p^2) = GF(p)[t]/(t^2 - %d), theta = %d + %d t, e0 = %d, |V0| = %d, classes %s, |S| = %d;"
        % (p, nr, th[0], th[1], e0, f, classes, int(inS.sum())))
    log("   n = %d vertices; the adjacency matrix is symmetric; the vectors theta^i are pairwise inequivalent modulo V0%s."
        % (n, " and represent all nonzero vectors" if complete else ""))
    log("   base graph G' = Q minus %d vertices: %d vertices, %d edges, %d triangles, degrees %s, no K4 (exhaustive check)."
        % (len(removed), len(V), nedges, ntri, ("%d" % degs[0]) if len(degs) == 1 else ("%d..%d" % (degs[0], degs[-1]))))
    log("   target graph G'' = G' minus %d further vertices: %d vertices (an induced subgraph, so no K4 either)."
        % (len(extra), len(V) - len(extra)))
    return {"n": n, "A": A, "V": V, "Vs": Vs, "adj": adj, "extra": extra, "nedges": nedges, "ntri": ntri, "removed": removed,
            "X": X, "vec_index": seen, "complete": complete, "p": p}


def build_system(C, G, log):
    """Checks the symmetry pi, the triangle list, D, the invariance and the budget inequality; returns the orbit data."""
    n = G["n"]; A = G["A"]; V = G["V"]; Vs = G["Vs"]; adj = G["adj"]; extra = G["extra"]
    if "step" in C:
        shift = (C["e0"] * C["step"]) % n
        assert shift != 0
        perm_list = [(i + shift) % n for i in range(n)]
        pi_text = "pi: i -> i + %d (mod %d)" % (shift, n)
    else:
        p = G["p"]; ma, mb, mc, md = (x % p for x in C["matrix"])
        assert G["complete"], "the vectors theta^i do not represent all nonzero vectors"
        perm_list = []
        for (x0, x1) in G["X"]:
            y0 = (ma * x0 + mc * x1) % p; y1 = (mb * x0 + md * x1) % p
            key = y0 * p + y1
            assert key in G["vec_index"], "the matrix maps a vertex to the zero vector"
            perm_list.append(G["vec_index"][key])
        pi_text = "pi: the map induced by (x0, x1) -> (%d x0 + %d x1, %d x0 + %d x1)" % (ma, mc, mb, md)
    assert sorted(perm_list) == list(range(n)), "pi is not a permutation of the vertices"
    assert any(perm_list[i] != i for i in range(n)), "pi is the identity"
    perm = np.array(perm_list, dtype=np.int64)
    assert bool((A[np.ix_(perm, perm)] == A).all()), "pi is not an automorphism of Q"
    assert all(int(perm[v]) in Vs for v in V), "the removed set is not invariant under pi"
    # pp[i][v] = pi^i(v), i = 0..q-1; q = order of pi
    pp = [list(range(n))]
    while True:
        nxt = [perm_list[v] for v in pp[-1]]
        if nxt == pp[0]: break
        pp.append(nxt)
        assert len(pp) <= 100000
    q = len(pp)
    assert q >= 2
    def ecode(u, v): return (u * n + v) if u < v else (v * n + u)
    if C["uniform"]:
        tri_list = []; seen_t = set()
        for u in V:
            for v in bits_above(adj[u], u):
                for x in bits_above(adj[u] & adj[v], v):
                    if (u, v, x) in seen_t: continue
                    for i in range(q):
                        seen_t.add(tuple(sorted((pp[i][u], pp[i][v], pp[i][x]))))
                    tri_list.append((u, v, x, Fraction(1)))
        C_tri = tri_list; C_D = []
    else:
        C_tri = C["tri"]; C_D = C["D"]
    den = 1
    for (_, _, _, w) in C_tri: den = den * w.denominator // math.gcd(den, w.denominator)
    for (_, _, d) in C_D: den = den * d.denominator // math.gcd(den, d.denominator)
    # ---- triangles: all images, distinct, triangles of G', weights >= 0
    seen = set()
    tri_rep = []                                    # (a, b, c, integer weight)
    for (a, b, c, w) in C_tri:
        wi = int(w * den)
        assert wi == w * den
        assert len({a, b, c}) == 3 and a in Vs and b in Vs and c in Vs
        assert (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[b] >> c) & 1, "listed triple is not a triangle of G'"
        for i in range(q):
            t = tuple(sorted((pp[i][a], pp[i][b], pp[i][c])))
            assert t not in seen, "a triangle is listed twice (or a triangle orbit is shorter than q)"
            seen.add(t)
        tri_rep.append((a, b, c, wi))
    ntri_listed = len(seen)
    if C["uniform"]: assert ntri_listed == G["ntri"], "uniform: not all triangles of G' were found"
    # ---- the index set I and D
    edge_idx = {}                                   # edge code -> (orbit number, offset i): edge = pi^i(representative)
    reps = []; dint = []
    if C_D:
        for (u, v, d) in C_D:
            assert d > 0 and u != v and u in Vs and v in Vs and (adj[u] >> v) & 1, "D-line is not an edge of G' or d <= 0"
            for i in range(q):
                code = ecode(pp[i][u], pp[i][v])
                assert code not in edge_idx, "an edge is listed twice (or an edge orbit is shorter than q)"
                edge_idx[code] = (len(reps), i)
            reps.append((u, v)); dint.append(int(d * den))
        for (a, b, c, wi) in tri_rep:
            if wi != 0:
                for (u, v) in ((a, b), (a, c), (b, c)):
                    assert ecode(u, v) in edge_idx, "an edge of a weighted triangle has no D entry"
    else:
        # D_e = sum of w_T over ALL listed triangles T containing e
        full = {}
        for (a, b, c, wi) in tri_rep:
            if wi == 0: continue
            for i in range(q):
                x, y, z = pp[i][a], pp[i][b], pp[i][c]
                for (u, v) in ((x, y), (x, z), (y, z)):
                    code = ecode(u, v)
                    full[code] = full.get(code, 0) + wi
        for code in sorted(full):
            if code in edge_idx: continue
            u, v = divmod(code, n)
            for i in range(q):
                c2 = ecode(pp[i][u], pp[i][v])
                assert c2 not in edge_idx and full[c2] == full[code], "edge orbit shorter than q or weights not invariant"
                edge_idx[c2] = (len(reps), i)
            assert full[code] > 0, "an edge has D_e <= 0 (give D-lines explicitly)"
            reps.append((u, v)); dint.append(full[code])
        assert len(edge_idx) == len(full)
    m = len(reps)
    log("2. symmetry: %s is an automorphism of Q that maps G' to itself; its order is q = %d." % (pi_text, q))
    nneg = sum(1 for t in tri_rep if t[3] < 0)
    log("   certificate: %d triangle orbits (%d triangles) with rational weights (%d orbits have a negative weight)%s, %d edge orbits (%d edges) with D > 0,"
        % (len(tri_rep), ntri_listed, nneg, " (uniform: all triangles of G', weight 1)" if C["uniform"] else "", m, m * q))
    log("   all orbits under pi have full length q; every listed triple is a triangle of G';")
    log("   common denominator of the numbers in the file: %d." % den)
    # ---- budget inequality (ii), exact
    isx = [False] * n
    for v in extra: isx[v] = True
    W2 = 0; nt2 = 0
    for (a, b, c, wi) in tri_rep:
        cnt = sum(1 for i in range(q) if not (isx[pp[i][a]] or isx[pp[i][b]] or isx[pp[i][c]]))
        W2 += wi * cnt; nt2 += cnt
    D2 = 0; ne2 = 0
    for (u, v), d in zip(reps, dint):
        cnt = sum(1 for i in range(q) if not (isx[pp[i][u]] or isx[pp[i][v]]))
        D2 += d * cnt; ne2 += cnt
    assert ne2 > 0 and D2 > 0
    budget_ok = D2 <= 3 * W2
    log("3. budget on G'': sum of D_e over its %d indexed edges = %d/%d;  3 * weight of its %d listed triangles = %d/%d;"
        % (ne2, D2, den, nt2, 3 * W2, den))
    log("   ratio = %.12f  (must be <= 1; exact integer comparison): %s" % (D2 / (3 * W2), "ok" if budget_ok else "VIOLATED"))
    assert budget_ok, "budget inequality violated"
    # ---- the integer matrices A_j: coef[(a,b)][j] = A[e_a, pi^j e_b], A = 3 M + 2 D (times den)
    coef = {}
    for a in range(m):
        coef[(a, a)] = {0: 2 * dint[a]}
    for (x, y, z, wi) in tri_rep:
        if wi == 0: continue
        es = [edge_idx[ecode(x, y)], edge_idx[ecode(x, z)], edge_idx[ecode(y, z)]]
        for (a, i) in es:
            for (b, k) in es:
                if (a, i) == (b, k): continue
                j = (k - i) % q
                dct = coef.setdefault((a, b), {})
                assert j not in dct, "two listed triangles share a pair of edges (impossible in a graph)"
                dct[j] = 3 * wi
    for (a, b), dct in coef.items():                    # symmetry A_j[a,b] = A_{-j}[b,a]
        other = coef[(b, a)]
        assert len(other) == len(dct) and all(other.get((-j) % q) == val for j, val in dct.items())
    nnz = sum(len(d) for d in coef.values())
    log("4. matrix A = 3M + 2D: size %d, commutes with the permutation of order %d; %d orbit blocks of size %d,"
        % (m * q, q, q, m))
    log("   %d nonzero integer coefficients A_j[a,b]." % nnz)
    return {"n": n, "q": q, "m": m, "den": den, "coef": coef, "dint": dint,
            "ratio": Fraction(D2, 3 * W2), "target_edges": ne2, "target_triangles": nt2,
            "budget_lhs": D2, "budget_rhs": 3 * W2, "reps": reps}


# ----------------------------------------------------------------------------------------------------
#  the blocks
# ----------------------------------------------------------------------------------------------------
_G = {}


def block_matrix(r):
    """The integer matrix Rhat of step (c) for the block B_r of A, and the bound dlt for its error."""
    q, m, coef, kk, C, S = _G["q"], _G["m"], _G["coef"], _G["k"], _G["C"], _G["S"]
    real = (r == 0) or (2 * r == q)
    X = np.zeros((m, m), dtype=np.int64)
    Y = None if real else np.zeros((m, m), dtype=np.int64)
    rowerr = [0] * m                                    # integer upper bounds for the row sums of |Rreal - Rhat|
    half = 1 << (Q_BITS - 1)
    for (a, b), dct in coef.items():
        kab = kk[a] * kk[b]
        cx = 0; cy = 0; cabs = 0
        for j, val in dct.items():
            t = (r * j) % q
            cx += val * C[t]; cy += val * S[t]; cabs += abs(val)
        assert abs(kab * cx) < 2 ** (61 + Q_BITS)
        X[a, b] = (kab * cx + half) >> Q_BITS           # nearest integer to kab*cx / 2^Q (ties up)
        if real:
            # C[t] = +-2^F exactly and S[t] = 0 exactly for these r: rounding <= 1/2 only
            assert all(S[(r * j) % q] == 0 and abs(C[(r * j) % q]) == 2 ** F_BITS for j in dct)
            rowerr[a] += 1
        else:
            assert abs(kab * cy) < 2 ** (61 + Q_BITS)
            Y[a, b] = (kab * cy + half) >> Q_BITS
            # each of the two entries: rounding <= 1/2, trigonometric error <= kab * cabs / 2^Q
            rowerr[a] += 1 + 2 * (-((-kab * cabs) >> Q_BITS))
    if real:
        assert bool((X == X.T).all())
        Rhat = X
    else:
        # ties-up rounding is not symmetric under negation; make Y exactly antisymmetric and account for it
        Ya = np.tril(Y, -1); Ya = Ya - Ya.T
        assert int(np.abs(Y - Ya).max()) <= 1
        rowerr = [e + int(c) for e, c in zip(rowerr, np.abs(Y - Ya).sum(axis=1))]
        Y = Ya
        assert bool((X == X.T).all()) and bool((Y == -Y.T).all())
        Rhat = np.block([[X, -Y], [Y, X]])
    dlt = max(rowerr)
    return Rhat, dlt


def block_job(r):
    t0 = time.time()
    Rhat, dlt = block_matrix(r)
    lb, info = certify_pd(Rhat, _G["blas"])
    res = {"r": r, "size": int(Rhat.shape[0]), "dlt": int(dlt), "seconds": round(time.time() - t0, 1)}
    res.update(info)
    res["lower_bound"] = None if lb is None else int(lb - dlt)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate"); ap.add_argument("--blas-products", action="store_true"); ap.add_argument("--json")
    args = ap.parse_args()
    t0 = time.time()
    lines = []
    def log(s):
        print(s); sys.stdout.flush(); lines.append(s)
    sha = hashlib.sha256(open(args.certificate, "rb").read()).hexdigest()
    log("verify_q2.py: certificate %s" % os.path.basename(args.certificate))
    log("sha256 %s" % sha)
    st = self_test()
    log("0. self-test of the positive-definiteness check (K_6 accepted, K_5 not accepted): ok")
    C = parse_certificate(args.certificate)
    G = build_graph(C, log)
    sysd = build_system(C, G, log)
    q, m = sysd["q"], sysd["m"]
    assert 2 * m <= 4096, "blocks too large for certify_pd"
    Ct, St = trig_tables(q)
    # scaling integers (any positive integers are valid)
    kk = [max(1, int(round(2.0 ** G_BITS / math.sqrt(sysd["coef"][(a, a)][0])))) for a in range(m)]
    _G.update({"q": q, "m": m, "coef": sysd["coef"], "k": kk, "C": Ct, "S": St, "blas": args.blas_products})
    log("5. positive definiteness of A, block by block (r = 0..%d; the blocks r and q-r are complex conjugates):" % (q // 2))
    results = []; ok = True; worst = None
    for r in range(q // 2 + 1):
        res = block_job(r); results.append(res)
        lbv = res["lower_bound"]
        good = lbv is not None and lbv > 0
        ok = ok and good
        if lbv is not None and (worst is None or lbv < worst): worst = lbv
        log("   block %3d: size %4d, float lambda_min %.6e, proved lower bound %s / 2^%d  %s  [%.1fs]"
            % (r, res["size"], res.get("float_lambda_min", float("nan")), lbv, UNIT_BITS, "ok" if good else "NO PROOF", res["seconds"]))
    nv = len(G["V"]) - len(G["extra"])
    if ok:
        log("   all %d blocks are positive definite (smallest proved bound %.6e in units where the diagonal is about 1)." % (q // 2 + 1, worst / 2.0 ** UNIT_BITS))
        log("6. conditions (i) and (ii) of the Lemma hold, and G'' has no K4.")
        log("THEOREM CHECKED: the graph G'' on %d vertices is K4-free and every 2-colouring of its edges contains a" % nv)
        log("monochromatic triangle.  Hence F_e(3,3;4) <= %d.   [%.0f s]" % (nv, time.time() - t0))
    else:
        log("NO PROOF: some block could not be certified.")
    if args.json:
        json.dump({"certificate": os.path.basename(args.certificate), "sha256": sha, "vertices": nv, "checked": ok,
                   "q": q, "m": m, "budget_ratio": float(sysd["ratio"]), "budget_lhs": str(sysd["budget_lhs"]), "budget_rhs": str(sysd["budget_rhs"]),
                   "den": str(sysd["den"]), "edges_base": G["nedges"], "triangles_base": G["ntri"], "target_edges": sysd["target_edges"],
                   "target_triangles": sysd["target_triangles"], "blocks": results, "mode": "blas" if args.blas_products else "integer",
                   "seconds": round(time.time() - t0, 1)}, open(args.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
