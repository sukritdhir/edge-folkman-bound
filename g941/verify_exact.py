#!/usr/bin/env python3
"""
verify_exact.py -- rigorous verifier for weighted spectral certificates of upper bounds on the
edge Folkman number F_e(3,3;4).

WHAT IS PROVED.  The certificate file describes a graph G'' and data (w, D).  If this program ends with
"THEOREM CHECKED", then G'' contains no K_4 and every colouring of the edges of G'' with two colours
contains a monochromatic triangle; hence F_e(3,3;4) <= |V(G'')|.

EVERY DECISIVE STEP USES EXACT INTEGER ARITHMETIC (Python integers and 64-bit integers with explicit
overflow guards).  Floating point is used only to *guess* auxiliary quantities (scaling factors, shifts,
approximate Cholesky factors, an approximate eigenspace, an approximate solution of a linear system).
A wrong guess can make the program fail, never make it accept.

THE CERTIFICATE (text file).
    n, S           base circulant Cay(Z_n, S)
    removed        vertices deleted from it: this gives the BASE graph G'
    extra          (optional) further vertices X: the TARGET graph is G'' = G' - X
    h              a multiplier: v -> h*v mod n maps G', the weights and D to themselves; q = order of h
    a b c w        weight w >= 0 of the triangle {a,b,c} of G' and of its q images under v -> h^i v
    D u v d        (optional) a number d > 0 for the edge {u,v} and its q images.  Without D-lines,
                   D_e := sum of w_T over the listed triangles T containing e.
All decimal numbers in the file are read as exact rationals.

THE CRITERION (Lemma).  Let I be the set of edges e of G' that carry a number D_e > 0; every edge of a
listed triangle of positive weight must lie in I.  For e != f in I put M[e,f] = w_T if e and f are two
edges of a listed triangle T (two edges lie in at most one triangle), and M[e,f] = 0 otherwise; M[e,e] = 0.
Let A = 3 M + 2 D  (D = diag(D_e); rows and columns indexed by I), and let P be the set of the edges in I
with both ends outside X (the indexed edges of G'').  If
    (i)  the principal submatrix A_PP of A (rows and columns in P) is positive definite, and
    (ii) sum of D_e over e in P   <=   3 * sum of w_T over the listed triangles T disjoint from X,
         and P is not empty,
then every 2-colouring of E(G'') has a monochromatic triangle.
Proof.  Let x in {+1,-1}^E(G'') be a colouring without monochromatic triangle, and extend it by 0 to the
other edges of G'.  A listed triangle T disjoint from X is a triangle of G'' (G'' is an induced subgraph);
its three edges do not all have the same colour, so the three pairwise products of their x-values sum
to -1.  A listed triangle that meets X has at most one edge in G'', and all its pairwise products vanish.
Hence x^T M x = -2 W'', where W'' is the sum of w_T over the listed triangles T disjoint from X.  Also
x^T D x = sum of D_e over e in P, which is <= 3 W'' by (ii).  Therefore
x^T A x = 3 x^T M x + 2 x^T D x <= -6 W'' + 6 W'' = 0.  But x vanishes on the edges of I outside P and has no zero entry
on P, so x^T A x = x_P^T A_PP x_P > 0 by (i).  This is a contradiction.  QED

HOW (i) IS CHECKED RIGOROUSLY.  There are two methods.  METHOD A proves that A itself is positive
definite; then every principal submatrix of A is.  METHOD B is used when method A gives no proof and X is
not empty; it proves (i) for certificates whose matrix A is indefinite.

METHOD A.
 (a) Symmetry reduction.  The map pi: e -> h*e permutes I in orbits of full length q (checked) and A commutes
     with pi (checked: weights and D are constant on orbits by construction of the file format, and the
     triangle list is closed under pi).  Choose representatives e_1..e_m of the orbits and put
         A_j[a,b] = A[e_a, pi^j e_b]      (integer matrices after clearing the common denominator),
         B_r = sum_j A_j * omega^(r j),  omega = exp(2 pi i / q),  r = 0..q-1   (Hermitian m x m).
     The vectors u_{r,a} = q^(-1/2) sum_j omega^(-r j) [pi^j e_a] form an orthonormal basis of C^I, and
     A u_{r,b} = sum_a conj(B_r[a,b]) u_{r,a}.  So A is unitarily similar to the direct sum of the matrices
     conj(B_r), and A is positive definite iff every B_r is.  Since B_{q-r} = conj(B_r), r <= q/2 suffices.
     The same holds for every real symmetric matrix that commutes with pi; in particular its smallest
     eigenvalue is the minimum of the smallest eigenvalues of its blocks.
 (b) Real form.  Write B_r = X + iY (X symmetric, Y antisymmetric).  B_r is positive definite iff the real
     symmetric matrix R = [[X, -Y], [Y, X]] is (z = u + iv gives z^H B_r z = [u;v]^T R [u;v]); the two
     matrices have the same smallest eigenvalue.
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

METHOD B.  Notation: K is the diagonal matrix with the entry k_a of (c) at all edges of the orbit of e_a, and
Ahat = K A K (an integer matrix).  A_PP is positive definite iff Ahat_PP = K_P A_PP K_P is.
 Lemma B.  Let C be a real symmetric matrix with C >= sigma Id for a number sigma > 0 (that is, C - sigma Id
     is positive semidefinite), let V be a real matrix with the same rows, and Ahat = C - V V^T.  Let P be a
     set of rows, V_P the rows P of V, and C_PP the principal submatrix.  Let Z be ANY real matrix of the shape
     of V_P and put
         E = V_P - C_PP Z,        H = Z^T C_PP Z   ( = Z^T (V_P - E) ).
     If  lambda_max(H)^(1/2) + ||E||_F / sigma^(1/2) < 1  (Frobenius norm), then Ahat_PP is positive definite.
 Proof.  Let y be a nonzero vector indexed by P, and |y|_C = (y^T C_PP y)^(1/2) >= sigma^(1/2) |y|.  Then
     y^T Ahat_PP y = |y|_C^2 - |V_P^T y|^2  and  V_P^T y = Z^T C_PP y + E^T y.  By the Cauchy-Schwarz inequality
     for the inner product (z, y) -> z^T C_PP y, every unit vector u satisfies
     u^T Z^T C_PP y <= |Z u|_C |y|_C = (u^T H u)^(1/2) |y|_C,  so  |Z^T C_PP y| <= lambda_max(H)^(1/2) |y|_C.
     Moreover |E^T y| <= ||E||_F |y| <= ||E||_F sigma^(-1/2) |y|_C.  So |V_P^T y| <= t |y|_C with
     t = lambda_max(H)^(1/2) + ||E||_F sigma^(-1/2) < 1,  and  y^T Ahat_PP y >= (1 - t^2) |y|_C^2 > 0.  QED
     (With Z = C_PP^-1 V_P one has E = 0 and H = V_P^T C_PP^-1 V_P, and Ahat_PP is positive definite iff
     lambda_max(H) < 1.  The lemma makes it unnecessary to trust the computed Z.)
 (e) The lift.  From floating point we take an integer array Vg, which defines an integer matrix Vhat with
     rows I and a few hundred columns (c, l), l = 0..q-1:  Vhat[pi^i e_a, (c,l)] = Vg[a, c, (i-l) mod q].
     Put V = Vhat / 2^S and C = Ahat + V V^T.  By its definition V V^T commutes with pi, and so does C.
     (The guess: V V^T is close to the matrix that moves the eigenvalues of Ahat below a cut-off kappa > 0
     up to kappa.)
 (f) C >= sigma Id is proved as in (a)-(d).  The block number r of Vhat Vhat^T is conj(v_r) v_r^T with
     v_r[a,c] = sum_t Vg[a,c,t] omega^(r t)  (a direct computation from the definition in (a)), so the block of
     C is  K B_r K + conj(v_r) v_r^T / 4^S.  Its real form, times 2^(F-Q), is enclosed by the integer matrix
     Rhat of (c) plus an integer matrix computed in `lift_block`, with a second explicit error bound.  sigma is
     the smallest of the lower bounds of the blocks, divided by 2^(F-Q).
 (g) From floating point we take an integer matrix Zhat (Zhat / 2^Z_BITS is close to the solution of
     Cbar_PP x = Vbar_P in coordinates where the diagonal of A is 1), and put Z = Zhat / 2^T.
 (h) Exact part.  The integer matrices
         Ehat = 2^(S+T) Vhat_P - (4^S K_P A_PP K_P + Vhat_P Vhat_P^T) Zhat      ( = 2^(2S+T) E ),
         Hn   = 2^(S+T) Zhat^T Vhat_P - Zhat^T Ehat                              ( = 4^(S+T) H )
     and the sum of the squares of the entries of Ehat are computed with Python integers (the large matrix
     products by splitting the integers into small digits, see `xmatmul`).  Then lambda_max(H) < h^2 is
     proved for a rational number h < 1 by applying (d) to an integer matrix that encloses 2^60 (h^2 Id - H),
     and ||E||_F <= (1 - h) sigma^(1/2) / 2 is an exact comparison of integers.  So
     lambda_max(H)^(1/2) + ||E||_F / sigma^(1/2) < h + (1 - h)/2 < 1, and Lemma B applies.

Usage:  python3 verify_exact.py certificate.txt [--jobs N] [--blas-products] [--json out.json]
Needs Python >= 3.8 and numpy.  Run time on 4 cores: about 3 minutes for a certificate of method A (integer
matrix products of size 2410) and about 6 minutes for the 734-vertex certificate of method B; with
--blas-products about 30 seconds and 2 minutes (see the comment at `matprod`).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, json, hashlib, argparse, itertools, math
from fractions import Fraction
import numpy as np

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
    # K_6: rounding of the entries moves eigenvalues by at most ne/2 units, far less than the bound obtained
    assert out[0] is not None and out[0] > 2 ** 40, out
    assert out[1] is None or out[1] <= 15, out
    return out


# ----------------------------------------------------------------------------------------------------
#  certificate
# ----------------------------------------------------------------------------------------------------
def parse_certificate(fn):
    C = {"tri": [], "D": []}
    with open(fn) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            key, *rest = line.split()
            if key in ("n", "h", "orbits", "edge_orbits"): C[key] = int(rest[0])
            elif key in ("S", "removed", "extra"): C[key] = [int(x) for x in rest]
            elif key == "D": C["D"].append((int(rest[0]), int(rest[1]), Fraction(rest[2])))
            else: C["tri"].append((int(key), int(rest[0]), int(rest[1]), Fraction(rest[2])))
    for key in ("n", "h", "S", "removed"):
        assert key in C, "certificate lacks the line '%s'" % key
    C.setdefault("extra", [])
    if "orbits" in C: assert len(C["tri"]) == C["orbits"]
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


def check_graph(C, log):
    n = C["n"]; S = set(C["S"]); removed = set(C["removed"]); extra = set(C["extra"])
    assert len(S) == len(C["S"]) and len(removed) == len(C["removed"]) and len(extra) == len(C["extra"])
    assert 0 not in S and all(0 < s < n for s in S) and all((n - s) % n in S for s in S), "S must be symmetric, 0 not in S"
    assert all(0 <= v < n for v in removed | extra) and not (removed & extra)
    V = [v for v in range(n) if v not in removed]; Vs = set(V)
    adj = {}
    for v in V:
        m = 0
        for s in S:
            u = (v + s) % n
            if u in Vs: m |= 1 << u
        adj[v] = m
    nedges = 0; ntri = 0
    for u in V:
        for v in bits_above(adj[u], u):
            nedges += 1
            common = adj[u] & adj[v]
            for x in bits_above(common, v):
                ntri += 1
                assert (common & adj[x]) >> (x + 1) == 0, "the base graph contains a K4 on %d,%d,%d" % (u, v, x)
    # informational: is S the set of nonzero k-th powers modulo a prime n?
    desc = ""
    if all(n % p for p in range(2, int(n ** 0.5) + 1)) and (n - 1) % len(S) == 0:
        k = (n - 1) // len(S)
        if S == set(pow(x, k, n) for x in range(1, n)):
            desc = " (S = the nonzero %d-th powers modulo the prime %d)" % (k, n)
    log("1. base graph G' = Cay(Z_%d, S) minus %d vertices, |S| = %d%s:" % (n, len(removed), len(S), desc))
    log("   %d vertices, %d edges, %d triangles, no K4 (exhaustive check)." % (len(V), nedges, ntri))
    log("   target graph G'' = G' minus %d further vertices: %d vertices (an induced subgraph, so no K4 either)."
        % (len(extra), len(V) - len(extra)))
    return V, Vs, adj, extra, nedges, ntri


def mult_order(h, n):
    assert math.gcd(h, n) == 1
    q = 1; x = h % n
    while x != 1:
        x = (x * h) % n; q += 1
    return q


def build_system(C, V, Vs, adj, extra, log):
    """Checks the triangle list, D, the invariance and the budget inequality; returns the orbit data."""
    n = C["n"]; h = C["h"] % n; S = set(C["S"])
    q = mult_order(h, n)
    assert all((h * v) % n in Vs for v in V), "vertex set not invariant under v -> h v"
    assert set((h * s) % n for s in S) == S, "S not invariant under multiplication by h"
    hp = [pow(h, i, n) for i in range(q)]
    # common denominator of all numbers
    den = 1
    for (_, _, _, w) in C["tri"]: den = den * w.denominator // math.gcd(den, w.denominator)
    for (_, _, d) in C["D"]: den = den * d.denominator // math.gcd(den, d.denominator)
    def ecode(u, v): return (u * n + v) if u < v else (v * n + u)
    # ---- triangles: all images, distinct, triangles of G', weights >= 0
    seen = set()
    tri_rep = []                                    # (a, b, c, integer weight)
    for (a, b, c, w) in C["tri"]:
        assert w >= 0
        wi = int(w * den)
        assert len({a, b, c}) == 3 and a in Vs and b in Vs and c in Vs
        assert (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[b] >> c) & 1, "listed triple is not a triangle of G'"
        for i in range(q):
            t = tuple(sorted(((a * hp[i]) % n, (b * hp[i]) % n, (c * hp[i]) % n)))
            assert t not in seen, "a triangle is listed twice (or a triangle orbit is shorter than q)"
            seen.add(t)
        tri_rep.append((a, b, c, wi))
    ntri_listed = len(seen)
    # ---- the index set I and D
    edge_idx = {}                                   # edge code -> (orbit number, offset i): edge = h^i * representative
    reps = []; dint = []
    if C["D"]:
        for (u, v, d) in C["D"]:
            assert d > 0 and u != v and u in Vs and v in Vs and (adj[u] >> v) & 1, "D-line is not an edge of G' or d <= 0"
            for i in range(q):
                code = ecode((u * hp[i]) % n, (v * hp[i]) % n)
                assert code not in edge_idx, "an edge is listed twice (or an edge orbit is shorter than q)"
                edge_idx[code] = (len(reps), i)
            reps.append((u, v)); dint.append(int(d * den))
        for (a, b, c, wi) in tri_rep:
            if wi > 0:
                for (u, v) in ((a, b), (a, c), (b, c)):
                    assert ecode(u, v) in edge_idx, "an edge of a weighted triangle has no D entry"
    else:
        # D_e = sum of w_T over ALL listed triangles T containing e
        full = {}
        for (a, b, c, wi) in tri_rep:
            if wi == 0: continue
            for i in range(q):
                x, y, z = (a * hp[i]) % n, (b * hp[i]) % n, (c * hp[i]) % n
                for (u, v) in ((x, y), (x, z), (y, z)):
                    code = ecode(u, v)
                    full[code] = full.get(code, 0) + wi
        for code in sorted(full):
            if code in edge_idx: continue
            u, v = divmod(code, n)
            for i in range(q):
                c2 = ecode((u * hp[i]) % n, (v * hp[i]) % n)
                assert c2 not in edge_idx and full[c2] == full[code], "edge orbit shorter than q or weights not invariant"
                edge_idx[c2] = (len(reps), i)
            reps.append((u, v)); dint.append(full[code])
        assert len(edge_idx) == len(full)
    m = len(reps)
    log("2. certificate: %d triangle orbits (%d triangles) with weights >= 0, %d edge orbits (%d edges) with D > 0,"
        % (len(tri_rep), ntri_listed, m, m * q))
    log("   all orbits under v -> %d v have full length q = %d; every listed triple is a triangle of G';" % (h, q))
    log("   common denominator of the numbers in the file: %d." % den)
    # ---- budget inequality (ii), exact
    isx = [False] * n
    for v in extra: isx[v] = True
    W2 = 0; nt2 = 0
    for (a, b, c, wi) in tri_rep:
        cnt = sum(1 for i in range(q) if not (isx[(a * hp[i]) % n] or isx[(b * hp[i]) % n] or isx[(c * hp[i]) % n]))
        W2 += wi * cnt; nt2 += cnt
    D2 = 0; ne2 = 0
    for (u, v), d in zip(reps, dint):
        cnt = sum(1 for i in range(q) if not (isx[(u * hp[i]) % n] or isx[(v * hp[i]) % n]))
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
    return {"n": n, "h": h, "q": q, "m": m, "den": den, "coef": coef, "dint": dint,
            "ratio": Fraction(D2, 3 * W2), "target_edges": ne2, "target_triangles": nt2,
            "budget_lhs": D2, "budget_rhs": 3 * W2, "reps": reps, "hp": hp}


# ----------------------------------------------------------------------------------------------------
#  the blocks
# ----------------------------------------------------------------------------------------------------
_G = {}


def block_matrix(r):
    """The integer matrix Rhat of step (c) for the block B_r of A, and the bound dlt for its error."""
    q, m, coef, kk, C, S = _G["q"], _G["m"], _G["coef"], _G["k"], _G["C"], _G["S"]
    real = (r == 0)
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
        X[a, b] = (kab * cx + half) >> Q_BITS           # nearest integer to kab*cx / 2^Q (ties up)
        if real:
            rowerr[a] += 1                              # rounding <= 1/2, no trigonometric error (C[0] = 2^F exactly)
        else:
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


# ----------------------------------------------------------------------------------------------------
#  method B: positive definiteness of the principal submatrix of A on the edges of the target graph
# ----------------------------------------------------------------------------------------------------
S_BITS = 4           # the lift is V V^T with V = Vhat / 2^S_BITS (in the scaled coordinates Ahat = K A K)
P_BITS = 16          # the Fourier coefficients of Vhat are rounded to integers ~ 2^P_BITS * (exact value)
Z_BITS = 30          # Zhat ~ 2^Z_BITS * (solution in normalised coordinates);  Z = Zhat / 2^T_BITS
T_BITS = G_BITS + Z_BITS
CHUNK = 16           # columns of Z that are processed together


def coef_lists(coef):
    ia = []; ib = []; jj = []; vv = []
    for (a, b) in sorted(coef):
        for j in sorted(coef[(a, b)]):
            ia.append(a); ib.append(b); jj.append(j); vv.append(coef[(a, b)][j])
    return np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64), np.array(jj, dtype=np.int64), vv


# ---------------- untrusted floating-point guesses (a wrong guess can only make the program fail) ----------------
def _float_block(r):
    q, m = _G["q"], _G["m"]
    ia, ib, jj, sc = _G["fl"]
    B = np.zeros((m, m), dtype=np.complex128)
    np.add.at(B, (ia, ib), sc * np.exp(2j * np.pi * ((r * jj) % q) / q))
    return B


def _eig_job(r):
    B = _float_block(r)
    w, U = np.linalg.eigh(B.real if (r == 0 or 2 * r == _G["q"]) else B)
    keep = _G["keep"]
    return w[:keep].copy(), U[:, :keep].copy()


def guess_lift(q, m, cl, kk, pool_map, log):
    """UNTRUSTED.  Returns the integer array Vg (shape m x qc x q) that defines the lift: column (c, l) of Vhat has the
    entry Vg[a, c, (i - l) mod q] at the edge pi^i e_a."""
    ia, ib, jj, vv = cl
    kf = np.array(kk, dtype=np.float64) / 2.0 ** G_BITS
    _G["fl"] = (ia, ib, jj, np.array([float(v) for v in vv]) * kf[ia] * kf[ib])
    _G["keep"] = min(m, 160)
    nr = q // 2 + 1
    low = pool_map(_eig_job, list(range(nr)))
    lam_min = min(float(w[0]) for w, _ in low)
    allw = np.sort(np.concatenate([w for w, _ in low]))
    base = max(-lam_min, 1e-3)
    lo, hi = 0.5 * base, 1.25 * base
    kappa = None; gap = 0.0
    for x, y in zip(allw[:-1], allw[1:]):                      # cut-off in the widest gap of the spectrum in [lo, hi]
        mid = 0.5 * (x + y)
        if lo <= mid <= hi and y - x > gap:
            kappa, gap = float(mid), float(y - x)
    if kappa is None:
        kappa = hi
    cnt = [int((w < kappa).sum()) for w, _ in low]
    qc = max(cnt)
    if not (0 < qc and all(c < len(w) for c, (w, _) in zip(cnt, low))):
        return None, None
    scale = 2.0 ** (G_BITS + S_BITS)
    Vh = np.zeros((q, m, qc), dtype=np.complex128)
    for r in range(nr):
        w, U = low[r]
        idx = np.nonzero(w < kappa)[0]
        g = U[:, idx] * np.sqrt(kappa - w[idx])                # the lift of the block r is g g^H: these eigenvalues move to kappa
        Vh[r, :, :len(idx)] = scale * np.conj(g)               # v_r = 2^(G+S) conj(g)
        if 0 < r and 2 * r != q:
            Vh[q - r] = np.conj(Vh[r])
    Vt = np.fft.fft(Vh, axis=0).real / q                       # Vt[t] = (1/q) sum_r v_r omega^(-r t), a real array
    Vg = np.rint(Vt).astype(np.int64).transpose(1, 2, 0).copy()
    mult = [1 if (r == 0 or 2 * r == q) else 2 for r in range(nr)]
    total = sum(f * c for f, c in zip(mult, cnt))
    neg = sum(f * int((w < 0).sum()) for f, (w, _) in zip(mult, low))
    log("   (a) guess of the lift (floating point): the matrix K A K, whose diagonal is about 2^%d, has %d negative"
        % (2 * G_BITS, neg))
    log("       eigenvalues, the smallest is %.6f (unit: the diagonal); cut-off %.6f; %d eigenvalues are lifted,"
        % (lam_min, kappa, total))
    log("       at most %d in a block; V has %d columns." % (qc, qc * q))
    return Vg, {"float_lambda_min": lam_min, "float_negative_eigenvalues": neg, "cutoff": kappa, "lifted": total,
                "per_block": qc}


def guess_Z(q, m, Vg, inP, log):
    """UNTRUSTED.  Zhat ~ 2^Z_BITS * (Cbar_PP)^-1 Vbar_P, where Cbar = Abar + Vbar Vbar^T in normalised coordinates
    (Abar = Ahat / 2^(2G) has unit diagonal, Vbar = Vhat / 2^(G+S))."""
    nr = q // 2 + 1; qc = Vg.shape[1]; n = m * q; p = qc * q
    scale = 2.0 ** (G_BITS + S_BITS)
    vh = np.fft.ifft(Vg.astype(np.float64), axis=2) * q         # vh[a, c, r] = sum_t Vg[a,c,t] omega^(r t)
    Cm = []; Ci = []
    for r in range(nr):
        g = np.conj(vh[:, :, r]) / scale
        Cr = _float_block(r) + g @ g.conj().T
        Cm.append(Cr); Ci.append(np.linalg.inv(Cr))

    def apply(blocks, X):                                        # X real [m, q, k]  ->  (invariant matrix with these blocks) X
        Xh = np.fft.fft(X, axis=1)
        Yh = np.empty_like(Xh)
        for r in range(q):
            Br = blocks[r] if r < nr else np.conj(blocks[q - r])
            Yh[:, r, :] = Br @ Xh[:, r, :]
        return np.fft.ifft(Yh, axis=1).real

    Fidx = np.nonzero(~inP)[0]; Pidx = np.nonzero(inP)[0]
    aF, iF = Fidx // q, Fidx % q
    Re = np.stack([c.real for c in Ci], axis=2).reshape(m * m, nr)
    Im = np.stack([c.imag for c in Ci], axis=2).reshape(m * m, nr)
    rr = np.arange(nr)[:, None]; jj = np.arange(q)[None, :]
    wt = np.where((rr == 0) | (2 * rr == q), 1.0, 2.0) / q
    th = 2 * np.pi * ((rr * jj) % q) / q
    cinv = (Re @ (wt * np.cos(th)) + Im @ (wt * np.sin(th))).reshape(m, m, q)    # entry [e_a, pi^j e_b] of Cbar^-1
    del Re, Im
    T = np.empty((m, q, len(Fidx)))                              # the columns F of Cbar^-1
    for i in range(q):
        T[:, i, :] = cinv[:, aF, (iF - i) % q]
    del cinv
    T = T.reshape(n, len(Fidx))
    Sinv = np.linalg.inv(T[Fidx])
    idx = (np.arange(q)[:, None] - np.arange(q)[None, :]) % q
    Vbar = Vg[:, :, idx].transpose(0, 2, 1, 3).reshape(n, p) / scale

    def solveP(B):                                               # solution of Cbar_PP x = B_P, extended by 0
        Y = apply(Ci, B.reshape(m, q, -1)).reshape(n, -1)
        X = Y - T @ (Sinv @ Y[Fidx])
        X[Fidx] = 0.0
        return X

    x = solveP(Vbar)
    for it in range(3):
        R = Vbar - apply(Cm, x.reshape(m, q, p)).reshape(n, p)
        R[Fidx] = 0.0
        res = float(np.linalg.norm(R))
        if it == 2 or res < 1e-9:
            break
        x += solveP(R)
    Zbar = x[Pidx]
    Hf = Zbar.T @ (Vbar[Pidx] - R[Pidx])
    lamH = float(np.linalg.eigvalsh(0.5 * (Hf + Hf.T))[-1])
    log("   (c) guess of Z: floating-point residual %.1e, largest eigenvalue of H about %.6f." % (res, lamH))
    return np.rint(Zbar * 2.0 ** Z_BITS).astype(np.int64), {"float_residual": res, "float_lambda_max_H": lamH}


# ---------------- exact part ----------------
def lift_block(r):
    """Integer matrices XL, YL (m x m) and an integer dl such that the real form [[XL,-YL],[YL,XL]] differs from
    2^(F-Q) * (real form of the block  conj(v_r) v_r^T / 4^S  of the lift),  v_r[a,c] = sum_t Vg[a,c,t] omega^(r t),
    by a symmetric matrix whose maximal absolute row sum is at most dl.  Python integers throughout."""
    q, m, C, S, Vg = _G["q"], _G["m"], _G["C"], _G["S"], _G["Vg"]
    qc = Vg.shape[1]
    Vo = Vg.astype(object)
    ct = np.array([C[(r * t) % q] for t in range(q)], dtype=object)
    st = np.array([S[(r * t) % q] for t in range(q)], dtype=object)
    NR = (Vo * ct).sum(axis=2); NI = (Vo * st).sum(axis=2)       # |NR - 2^F Re v_r| <= eabs,  |NI - 2^F Im v_r| <= eabs
    eabs = int(np.abs(Vg).sum(axis=2).max())
    sh1 = F_BITS - P_BITS
    assert eabs < (1 << (sh1 - 1))
    half1 = 1 << (sh1 - 1)
    vR = (NR + half1) >> sh1; vI = (NI + half1) >> sh1           # |vR - 2^P Re v_r| <= 1/2 + eabs / 2^sh1 <= 1, same for vI
    Qm = vR.dot(vI.T)
    LR = vR.dot(vR.T) + vI.dot(vI.T)       # 4^P Re(conj(v_a) v_b) up to an error <= rho_a + rho_b + 2 qc  (see dl below)
    LI = Qm - Qm.T                         # 4^P Im(conj(v_a) v_b) up to the same error; exactly antisymmetric
    sh2 = 2 * S_BITS + 2 * P_BITS - (F_BITS - Q_BITS)
    assert sh2 >= 1
    half2 = 1 << (sh2 - 1)
    XL = (LR + half2) >> sh2
    low = np.tril((LI + half2) >> sh2, -1)
    YL = low - low.T
    assert bool((XL == XL.T).all())
    # |x y - x' y'| <= |x - x'| |y'| + |x'| |y - y'| + |x - x'| |y - y'|, so with rho_a = sum_c (|vR[a,c]| + |vI[a,c]|)
    # each of the two entries (a,b) has an error <= 1/2 (rounding) + (rho_a + rho_b + 2 qc) / 2^sh2,
    # and a row of the real form has 2m such entries
    rho = [int(x) for x in (np.abs(vR) + np.abs(vI)).sum(axis=1)]
    tot = sum(rho)
    dl = max(m + ((2 * (m * ra + tot + 2 * qc * m) + (1 << sh2) - 1) >> sh2) for ra in rho)
    assert max(int(np.abs(XL).max()), int(np.abs(YL).max())) < 2 ** 60
    return XL.astype(np.int64), YL.astype(np.int64), int(dl)


def block_job_lifted(r):
    t0 = time.time()
    Rhat, dlt = block_matrix(r)
    XL, YL, dl = lift_block(r)
    if r == 0:
        assert not YL.any()
        Rhat = Rhat + XL
    else:
        Rhat = Rhat + np.block([[XL, -YL], [YL, XL]])
    lb, info = certify_pd(Rhat, _G["blas"])
    res = {"r": r, "size": int(Rhat.shape[0]), "dlt": int(dlt), "dlt_lift": int(dl), "seconds": round(time.time() - t0, 1)}
    res.update(info)
    res["lower_bound"] = None if lb is None else int(lb - dlt - dl)
    return res


def digits(X, bits):
    """Balanced digits: X = sum_j D_j 2^(bits j) with -2^(bits-1) <= D_j < 2^(bits-1).  X: int64 array or array of
    Python integers; the digits are int64 arrays."""
    half = 1 << (bits - 1); mask = (1 << bits) - 1
    if X.dtype != object:
        assert X.dtype == np.int64 and (X.size == 0 or int(np.abs(X).max()) < 2 ** 62)
    out = []
    while True:
        d = ((X + half) & mask) - half
        out.append(d.astype(np.int64))
        X = (X - d) >> bits
        if not X.any():
            break
    return out


def xmatmul(DA, DB, bits, blas):
    """Exact product of the integer matrices sum_i DA_i 2^(bits i) and sum_j DB_j 2^(bits j), as Python integers.
    Every partial product DA_i @ DB_j has entries of absolute value <= inner * 2^(2 bits - 2).  Default: numpy's
    64-bit integer matrix product (no overflow by the assertion).  With blas=True the partial products are
    computed in double precision, which is exact because all partial sums are integers below 2^53."""
    inner = DA[0].shape[1]
    bound = inner << (2 * (bits - 1))
    assert min(len(DA), len(DB)) * bound < 2 ** 62
    if blas:
        assert bound < 2 ** 53
        DBf = [b.astype(np.float64) for b in DB]
    lev = [None] * (len(DA) + len(DB) - 1)
    for i, a in enumerate(DA):
        af = a.astype(np.float64) if blas else None
        for j, b in enumerate(DB):
            if blas:
                Pf = af @ DBf[j]
                assert float(np.abs(Pf).max()) <= bound
                P = Pf.astype(np.int64)
            else:
                P = a @ b
            lev[i + j] = P if lev[i + j] is None else lev[i + j] + P
    res = None
    for k in range(len(lev) - 1, -1, -1):
        t = lev[k].astype(object)
        res = t if res is None else (res << bits) + t
    return res


def full_sparse(q, m, cl):
    """The integer matrix A (order m q; the edge pi^i e_a has the index a q + i) in row-sorted coordinate form."""
    ia, ib, jj, vv = cl
    assert all(abs(v) < 2 ** 40 for v in vv)
    i = np.arange(q, dtype=np.int64)
    rows = (ia[:, None] * q + i[None, :]).ravel()
    cols = (ib[:, None] * q + (i[None, :] + jj[:, None]) % q).ravel()
    vals = np.repeat(np.array(vv, dtype=np.int64), q)
    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]
    ptr = np.searchsorted(rows, np.arange(m * q))
    assert bool((np.diff(np.append(ptr, len(rows))) > 0).all())
    rowabs = int(np.add.reduceat(np.abs(vals), ptr).max())
    return cols, vals, ptr, rowabs


def chunk_job(c0):
    """Columns c0..c1-1 of  W = Vhat_P^T Zhat,  Ehat = 2^(S+T) Vhat_P - (4^S K_P A_PP K_P + Vhat_P Vhat_P^T) Zhat  and
    Zhat^T Ehat; returns W[:, c0:c1], (Zhat^T Ehat)[:, c0:c1] and the sum of the squares of these columns of Ehat."""
    X = _G["xb"]
    bits, blas, n, Pidx, kP = X["bits"], X["blas"], X["n"], X["Pidx"], X["kP"]
    cols, vals, ptr = X["sparse"]
    c1 = min(X["p"], c0 + CHUNK)
    Zc = X["Zhat"][:, c0:c1]
    Wc = xmatmul(X["DVT"], digits(Zc, bits), bits, blas)                 # Vhat_P^T Zhat
    KZ = Zc.astype(object) * kP                                           # K_P Zhat
    AY = None
    for d in reversed(digits(KZ, bits)):
        L = np.zeros((n, c1 - c0), dtype=np.int64); L[Pidx] = d          # extended by 0 outside P
        t = np.add.reduceat(vals[:, None] * L[cols], ptr, axis=0)[Pidx].astype(object)    # |.| <= rowabs 2^(bits-1) < 2^62
        AY = t if AY is None else (AY << bits) + t                        # = A_PP K_P Zhat  (exact)
    T2 = (AY * kP) << (2 * S_BITS)                                        # 4^S K_P A_PP K_P Zhat
    T3 = xmatmul(X["DV"], digits(Wc, bits), bits, blas)                   # Vhat_P (Vhat_P^T Zhat)
    T1 = X["VP"][:, c0:c1].astype(object) << (S_BITS + T_BITS)
    E = T1 - T2 - T3
    return c0, Wc, xmatmul(X["DZT"], digits(E, bits), bits, blas), int((E * E).sum())


def exact_subspace_check(q, m, cl, kk, Vg, Zhat, inP, sigma_hat, blas, pool_map, log):
    """EXACT INTEGER ARITHMETIC.  Notation: Ahat = K A K;  Chat = Ahat + Vhat Vhat^T / 4^S, for which
    Chat >= (sigma_hat / 2^(F-Q)) Id has been proved;  V = Vhat_P / 2^S (rows P of Vhat);  Z = Zhat / 2^T.
    Computes E = V - Chat_PP Z and H = Z^T Chat_PP Z = Z^T (V - E) exactly and checks
        lambda_max(H) < h^2   and   ||E||_F <= (1 - h) sigma^(1/2) / kq     (sigma = sigma_hat / 2^(F-Q), kq >= 2).
    Then Ahat_PP = Chat_PP - V V^T is positive definite (lemma in the header)."""
    t0 = time.time()
    n = m * q; qc = Vg.shape[1]; p = qc * q
    Pidx = np.nonzero(inP)[0]; nP = len(Pidx)
    assert Zhat.shape == (nP, p) and Zhat.dtype == np.int64 and Vg.dtype == np.int64 and sigma_hat > 0 and nP > 0
    assert int(np.abs(Zhat).max()) < 2 ** 60 and int(np.abs(Vg).max()) < 2 ** 40 and p <= 4096
    idx = (np.arange(q)[:, None] - np.arange(q)[None, :]) % q
    VP = Vg[:, :, idx].transpose(0, 2, 1, 3).reshape(n, p)[Pidx]         # Vhat[pi^i e_a, (c,l)] = Vg[a, c, (i-l) mod q]
    bits = 16 if blas else 22
    cols, vals, ptr, rowabs = full_sparse(q, m, cl)
    assert (rowabs << (bits - 1)) < 2 ** 62
    DV = digits(VP, bits)
    _G["xb"] = {"bits": bits, "blas": blas, "n": n, "p": p, "Pidx": Pidx, "VP": VP, "Zhat": Zhat, "sparse": (cols, vals, ptr),
                "kP": np.array([kk[int(e) // q] for e in Pidx], dtype=object)[:, None],
                "DV": DV, "DVT": [np.ascontiguousarray(d.T) for d in DV],
                "DZT": [np.ascontiguousarray(d.T) for d in digits(Zhat, bits)]}
    W = np.empty((p, p), dtype=object); ZE = np.empty((p, p), dtype=object)
    e2 = 0
    for (c0, Wc, ZEc, e2c) in pool_map(chunk_job, list(range(0, p, CHUNK))):
        W[:, c0:c0 + Wc.shape[1]] = Wc; ZE[:, c0:c0 + Wc.shape[1]] = ZEc; e2 += e2c
    del _G["xb"]
    Hn = (W.T << (S_BITS + T_BITS)) - ZE                                 # H = Hn / 4^(S+T)
    assert bool((Hn == Hn.T).all()), "H is not symmetric"
    sh = 2 * (S_BITS + T_BITS)
    info = {"columns": p, "rows": nP, "E_squared_norm_numerator": str(e2), "seconds_exact": round(time.time() - t0, 1)}
    # ---- lambda_max(H) < h^2 with h = hn / 2^20; the choice of h is a floating-point guess
    lam = float(np.linalg.eigvalsh(np.array([[float(x) for x in row] for row in Hn]) / 2.0 ** sh)[-1])
    info["float_lambda_max_H"] = lam
    if not lam < 1:
        return False, info
    hd = 1 << 20
    hn = int(math.ceil(math.sqrt(lam + 0.02 * (1 - lam)) * hd))
    if not 0 < hn < hd:
        return False, info
    Rm = -((Hn << UNIT_BITS) >> sh)                                      # -floor(2^60 H)
    for i in range(p):
        Rm[i, i] += (hn * hn) << (UNIT_BITS - 40)                        # 2^60 h^2
    # 0 <= Rm - 2^60 (h^2 Id - H) < 1 entrywise, so lambda_min(2^60 (h^2 Id - H)) >= lambda_min(Rm) - p
    assert max(abs(int(x)) for x in Rm.ravel()) < 2 ** 61
    lb, _ = certify_pd(Rm.astype(np.int64), blas)
    info.update({"h_numerator": hn, "h_denominator": hd, "H_lower_bound": None if lb is None else int(lb - p)})
    if lb is None or lb - p <= 0:
        return False, info
    # ---- ||E||_F <= (1 - h) sigma^(1/2) / kq   <=>   kq^2 e2 2^(F-Q) hd^2 <= (hd - hn)^2 16^S 4^T sigma_hat
    lhs = e2 * (1 << (F_BITS - Q_BITS)) * hd * hd
    rhs = (hd - hn) ** 2 * (1 << (4 * S_BITS + 2 * T_BITS)) * sigma_hat
    kq = 1 << 12
    while kq > 1 and kq * kq * lhs > rhs:
        kq >>= 1
    info["residual_quotient"] = kq
    return kq >= 2, info


def method_b(sysd, kk, extra, blas, pool_map, log):
    """Proves that the principal submatrix of A on the edges of the target graph is positive definite."""
    q, m, n, coef, reps, hp = sysd["q"], sysd["m"], sysd["n"], sysd["coef"], sysd["reps"], sysd["hp"]
    isx = [False] * n
    for v in extra: isx[v] = True
    inP = np.array([not (isx[(u * hp[i]) % n] or isx[(v * hp[i]) % n]) for (u, v) in reps for i in range(q)], dtype=bool)
    nP = int(inP.sum())
    assert nP == sysd["target_edges"]
    log("6. the principal submatrix of A on the %d indexed edges of G'' (%d indexed edges of G' meet the further vertices):"
        % (nP, len(inP) - nP))
    cl = coef_lists(coef)
    out = {"target_rows": nP}
    Vg, ginfo = guess_lift(q, m, cl, kk, pool_map, log)
    if Vg is None:
        log("   no suitable lift was found.")
        return False, out
    out["lift"] = ginfo
    _G["Vg"] = Vg
    log("   (b) the matrix Chat = K A K + V V^T, which commutes with the permutation: its %d blocks, exact integer residuals ..." % (q // 2 + 1))
    results = pool_map(block_job_lifted, list(range(q // 2 + 1)))
    out["blocks"] = results
    good = True
    for res in results:
        lb = res["lower_bound"]
        ok = lb is not None and lb > 0
        good = good and ok
        if "normE" in res:
            log("       r = %2d: size %4d, sigma = %d, ||E|| <= %d, enclosure errors <= %d + %d  ->  lambda_min >= %d  %s  [%.0fs]"
                % (res["r"], res["size"], res["sigma"], res["normE"], res["dlt"], res["dlt_lift"], lb,
                   "POSITIVE" if ok else "NO PROOF", res["seconds"]))
        else:
            log("       r = %2d: size %4d: NO PROOF (floating point: the smallest eigenvalue is about %.6f times the unit 2^%d)"
                % (res["r"], res["size"], res["float_lambda_min"], UNIT_BITS))
    if not good:
        log("   the lifted matrix was not certified.")
        return False, out
    sigma_hat = min(res["lower_bound"] for res in results)
    out["sigma_hat"] = sigma_hat
    log("       Chat >= sigma Id with sigma = %d / 2^%d, which is %.6f times the unit 2^%d of its diagonal."
        % (sigma_hat, F_BITS - Q_BITS, sigma_hat / 2.0 ** UNIT_BITS, 2 * G_BITS))
    Zhat, zinfo = guess_Z(q, m, Vg, inP, log)
    out["Z"] = zinfo
    ok, xinfo = exact_subspace_check(q, m, cl, kk, Vg, Zhat, inP, sigma_hat, blas, pool_map, log)
    out["exact"] = xinfo
    if "h_numerator" in xinfo and xinfo.get("H_lower_bound") is not None:
        log("   (d) exact: E = V - Chat_PP Z and H = Z^T Chat_PP Z with %d columns and %d rows;" % (xinfo["columns"], xinfo["rows"]))
        log("       lambda_max(H) < h^2 for h = %d/%d, since lambda_min(2^60 (h^2 Id - H)) >= %d;"
            % (xinfo["h_numerator"], xinfo["h_denominator"], xinfo["H_lower_bound"]))
    if not ok:
        log("   the inequality  lambda_max(H)^(1/2) + ||E||_F / sigma^(1/2) < 1  was not certified.")
        return False, out
    hn, hd, kq = xinfo["h_numerator"], xinfo["h_denominator"], xinfo["residual_quotient"]
    log("       ||E||_F <= (1 - h) sigma^(1/2) / %d.   Hence lambda_max(H)^(1/2) + ||E||_F / sigma^(1/2) < 1," % kq)
    log("       and the principal submatrix is positive definite.")
    # y^T Ahat_PP y >= (1 - (h + (1-h)/kq)^2) y^T Chat_PP y >= (1 - (h + (1-h)/kq)^2) sigma |y|^2
    t = Fraction(hn, hd) + Fraction(hd - hn, hd * kq)
    out["lambda_min_scaled"] = (1 - t * t) * Fraction(sigma_hat, 2 ** (F_BITS - Q_BITS))
    return True, out


def self_test_b(blas, pool_map):
    """Method B on a toy system (a test of the linear algebra only; the toy graph is not K_4-free): the complete
    graph on the nine vertices 1, 3, 9, 2, 6, 5, 4, 12, 10 modulo 13 with the symmetry v -> 3v of order 3; the
    27 triangle orbits of length 3, where the orbit with the smallest triangle {a,b,c} has the weight
    1 + (abc mod 3); D_e = c * (sum of the weights of the triangles through e); X = {1, 3, 5}.
    For c = 77/100 the matrix A has negative eigenvalues in the real block and in the complex blocks, while its
    principal submatrix on the 15 edges that avoid X is positive definite: method A must give no proof and
    method B must give one.  For c = 72/100 that principal submatrix has a negative eigenvalue, and method B
    must not give a proof."""
    n, q = 13, 3
    hp = [1, 3, 9]
    verts = [1, 2, 3, 4, 5, 6, 9, 10, 12]; X = {1, 3, 5}
    edge_idx = {}; reps = []
    for (u, v) in itertools.combinations(verts, 2):
        if (u, v) in edge_idx: continue
        for i in range(q):
            edge_idx[tuple(sorted(((u * hp[i]) % n, (v * hp[i]) % n)))] = (len(reps), i)
        reps.append((u, v))
    m = len(reps)
    tri = []; seen = set(); d0 = [0] * m                # (representative, weight); d0[a] = sum of the weights at e_a
    for T in itertools.combinations(verts, 3):
        orb = {tuple(sorted((v * f) % n for v in T)) for f in hp}
        if T in seen or len(orb) < q: continue
        seen |= orb
        w = 1 + (T[0] * T[1] * T[2]) % 3
        tri.append((T, w))
        for T2 in orb:
            for p in itertools.combinations(T2, 2):
                if edge_idx[p][1] == 0: d0[edge_idx[p][0]] += w
    assert m == 12 and len(edge_idx) == 36 and len(tri) == 27 and min(d0) > 0
    Ctab, Stab = trig_tables(q)
    saved = dict(_G)
    out = []
    for c100 in (77, 72):                               # all numbers are multiplied by 100
        coef = {(a, a): {0: 2 * c100 * d0[a]} for a in range(m)}
        for (T, w) in tri:
            es = [edge_idx[p] for p in itertools.combinations(T, 2)]
            for (a, i) in es:
                for (b, k) in es:
                    if (a, i) != (b, k):
                        coef.setdefault((a, b), {})[(k - i) % q] = 300 * w
        kk = [max(1, int(round(2.0 ** G_BITS / math.sqrt(coef[(a, a)][0])))) for a in range(m)]
        _G.clear()
        _G.update({"q": q, "m": m, "coef": coef, "k": kk, "C": Ctab, "S": Stab, "blas": blas})
        blocks = [block_job(r) for r in range(q // 2 + 1)]
        proved_a = all((res["lower_bound"] or 0) > 0 for res in blocks)
        sysd = {"q": q, "m": m, "n": n, "coef": coef, "reps": reps, "hp": hp, "target_edges": 15}
        proved_b, info = method_b(sysd, kk, X, blas, pool_map, lambda s: None)
        out.append((proved_a, proved_b, [res["float_lambda_min"] < 0 for res in blocks], info.get("lift", {}).get("lifted")))
    _G.clear(); _G.update(saved)
    assert [o[:2] for o in out] == [(False, True), (False, False)] and out[0][2] == [True, True], out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--blas-products", action="store_true", help="exact matrix products through double-precision BLAS (faster)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    t0 = time.time()
    def log(s): print(s, flush=True)
    sha = hashlib.sha256(open(args.certificate, "rb").read()).hexdigest()
    log("certificate %s   sha256 %s" % (os.path.basename(args.certificate), sha))
    def pool_map(fn, items):
        """fn(item) for all items, in the order of the items; the worker processes see the current contents of _G."""
        if args.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(args.jobs) as pool:
                return pool.map(fn, items, chunksize=1)
        return [fn(x) for x in items]

    self_test()
    self_test_b(args.blas_products, pool_map)
    log("0. self-tests: K_6 certified, K_5 not certified; method B on a toy matrix: a positive definite principal")
    log("   submatrix of an indefinite matrix certified, an indefinite principal submatrix not certified: ok")
    C = parse_certificate(args.certificate)
    V, Vs, adj, extra, nedges, ntri = check_graph(C, log)
    sysd = build_system(C, V, Vs, adj, extra, log)
    q, m = sysd["q"], sysd["m"]
    Ctab, Stab = trig_tables(q)
    kk = [max(1, int(round(2.0 ** G_BITS / math.sqrt(sysd["coef"][(a, a)][0])))) for a in range(m)]
    assert all(0 < k < 2 ** (G_BITS + 2) for k in kk)
    _G.update({"q": q, "m": m, "coef": sysd["coef"], "k": kk, "C": Ctab, "S": Stab, "blas": args.blas_products})

    rs = list(range(0, q // 2 + 1))
    log("5. positive definiteness of the %d blocks B_r, r = 0..%d (B_{q-r} is the complex conjugate of B_r);"
        % (len(rs), rs[-1]))
    log("   exact integer residuals, %s; %d processes ..." % ("products through BLAS" if args.blas_products else "integer matrix products", args.jobs))
    results = pool_map(block_job, rs)
    ok = True
    worst = None
    kmax = max(kk[a] ** 2 * sysd["coef"][(a, a)][0] for a in range(m))
    for res in results:
        lb = res["lower_bound"]
        good = lb is not None and lb > 0
        ok = ok and good
        if good:
            # lambda_min(N) + 2/3 >= (2/3) * lb * 2^(Q-F) / max_a (k_a^2 * A[e_a,e_a]),  N = D^-1/2 M D^-1/2
            mu = Fraction(2 * lb * 2 ** Q_BITS, 3 * 2 ** F_BITS * kmax)
            res["margin_lower_bound"] = float(mu)
            worst = mu if worst is None else min(worst, mu)
        if "normE" in res:
            log("   r = %2d: size %4d, sigma = %d, ||E|| <= %d, enclosure error <= %d  ->  lambda_min >= %d  %s  [%.0fs]"
                % (res["r"], res["size"], res["sigma"], res["normE"], res["dlt"], lb, "POSITIVE" if good else "NO PROOF", res["seconds"]))
        else:
            log("   r = %2d: size %4d: NO PROOF (floating point: the smallest eigenvalue is about %.6f times the unit 2^%d)"
                % (res["r"], res["size"], res["float_lambda_min"], UNIT_BITS))
    nv = len(V) - len(extra)
    method = "A"
    binfo = None
    if ok:
        log("   all blocks are positive definite.  Rigorous bound: lambda_min(D^-1/2 M D^-1/2) >= -2/3 + %.6f" % float(worst))
    elif extra:
        method = "B"
        worst = None
        log("   Method A gives no proof that A is positive definite.  Method B:")
        ok, binfo = method_b(sysd, kk, extra, args.blas_products, pool_map, log)
        if ok:
            # as in method A, with the principal submatrices D'', M'' on the edges of G'':
            # lambda_min(D''^-1/2 M'' D''^-1/2) + 2/3 >= (2/3) lambda_min(K_P A_PP K_P) / max_a (k_a^2 A[e_a,e_a])
            worst = Fraction(2, 3) * binfo["lambda_min_scaled"] / kmax
            log("   Rigorous bound: lambda_min(D''^-1/2 M'' D''^-1/2) >= -2/3 + %.6f  (D'', M'': rows and columns of the edges of G'')" % float(worst))
        binfo.pop("lambda_min_scaled", None)
    if ok:
        log("THEOREM CHECKED: the graph G'' on %d vertices contains no K4 and every 2-colouring of its edges has a" % nv)
        log("monochromatic triangle.  Hence F_e(3,3;4) <= %d.   [total time %.0fs]" % (nv, time.time() - t0))
    else:
        log("NO PROOF OBTAINED from this certificate.")
    if args.json:
        out = {"certificate": os.path.basename(args.certificate), "sha256": sha, "valid": ok, "vertices": nv, "method": method,
               "base_vertices": len(V), "base_edges": nedges, "base_triangles": ntri, "q": q, "block_size": m,
               "target_edges": sysd["target_edges"], "target_triangles": sysd["target_triangles"],
               "budget_ratio": float(sysd["ratio"]), "margin_lower_bound": None if worst is None else float(worst),
               "budget_lhs": str(sysd["budget_lhs"]), "budget_rhs": str(sysd["budget_rhs"]), "denominator": str(sysd["den"]),
               "extra": sorted(extra), "triangle_orbits": len(C["tri"]),
               "blocks": results, "method_b": binfo, "blas_products": args.blas_products,
               "seconds_total": round(time.time() - t0, 1)}
        json.dump(out, open(args.json, "w"), indent=1)
    return ok


if __name__ == "__main__":
    if not __debug__:
        sys.exit("Do not run this program with -O or PYTHONOPTIMIZE: its checks are assert statements.")
    try:
        ok = main()
    except AssertionError as exc:
        print("CERTIFICATE REJECTED, a check failed: %s" % (exc if str(exc) else "see the traceback below"), flush=True)
        if not str(exc):
            raise
        ok = False
    sys.exit(0 if ok else 1)
