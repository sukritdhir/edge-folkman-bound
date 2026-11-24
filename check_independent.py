#!/usr/bin/env python3
"""
check_independent.py -- a second, separately written check of a certificate for F_e(3,3;4) <= N
on a graph of the family Q(p, e) or on an induced subgraph of such a graph.

This program shares no code with verify_q2.py; the two programs have only the certificate file in common.
It is meant as a cross-check.  The differences:

  * THE GRAPH is built from its definition as a graph on vectors.  H is the group of the e-th powers
    modulo p.  The vertices of Q(p, e) are the classes of the nonzero vectors of GF(p)^2 under
    multiplication by the scalars of H, and two classes [x], [y] are adjacent iff det(x, y) lies in H.
    The numbers 0, ..., n-1 that the certificate uses for the vertices are translated by
    i -> the class of theta^i  (theta = a + b t in GF(p)[t]/(t^2 - nr), read as the vector (a, b)),
    and the program checks that this is a bijection onto the set of all classes.

  * NO SYMMETRY is used.  The weights of the certificate are written out for every triangle and the numbers
    D for every edge.  The certificate lists one member of each orbit of a permutation pi of the vertices:
    the shift i -> i + e0*step (line `step`), or the map of the classes induced by the linear map
    (x1, x2) -> (a x1 + c x2, b x1 + d x2) (line `matrix a b c d`).  The program applies pi again and again
    and checks that every image is a triangle or an edge of the base graph and that none occurs twice; it is
    NOT assumed that pi is an automorphism.  The inequality (ii) is summed in exact rational arithmetic, and
    the matrix A is handled as one matrix on all indexed edges (21666 rows and columns for the 483-vertex
    graph, 28078 for the 556-vertex graph).

  * POSITIVE DEFINITENESS is shown in floating-point arithmetic with an explicit error bound (verify_q2.py
    uses integers and the symmetry).  Let s_e = 1/sqrt(2 D_e) and N = S A S (S = diag(s_e)); N has diagonal
    1 and off-diagonal entries 3 w_T s_e s_f.  A is positive definite iff N is.  The program chooses a shift
    sigma > 0 (0.9 times an estimate of the smallest eigenvalue of N; the estimate is a guess and is not used
    in the proof), computes a Cholesky factor L of N - sigma*I in double precision, then the residual
    E = N - sigma*I - L L^T entry by entry, and the maximal absolute row sum of E.  For every unit vector v,
        v^T N v = sigma + |L^T v|^2 + v^T E v >= sigma - ||E||_inf.
    The floating-point errors made while computing E are bounded by the standard estimate for inner
    products (N. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Section 3.1: the computed
    value of an inner product of m terms differs from the exact one by at most g |x|^T |y|, g = m u/(1 - m u),
    u = 2^-53, for every order of evaluation):
        |computed E_ij - E_ij| <= g |l_i| |l_j| + (rounding of one subtraction),
    so the row sums of |E| exceed the computed ones by at most g * m * max|l_i|^2 (plus terms below 1e-12).
    The entries of the floating-point matrix differ from those of S A S (S the diagonal matrix of the
    floating-point numbers s_e, which may be any positive numbers) by at most 4 u per entry; this is covered
    by the allowance 1e-12.  The program prints
        bound = sigma - ||E||_inf (computed) * (1 + 1e-9) - g * m * max|l_i|^2 * 1.001 - 1e-12
    and accepts iff bound > 0.

What is checked, in this order:
  1. H contains -1 and is sum-free (no a + b = c with a, b, c in H); the classes; the translation of the
     vertex numbers; degrees; the number of triangles on every edge of Q(p, e).
  2. The base graph G' (Q minus the `removed` vertices) contains no K4: for every edge, the common
     neighbours of its ends are pairwise non-adjacent (exhaustive).  The target graph G'' is G' minus the
     `extra` vertices; it is an induced subgraph of G', so it contains no K4 either.
  3. The certificate: every listed triangle and all its images are triangles of G', every D-line and all
     its images are edges of G', no triangle and no edge occurs twice, every edge of a triangle of nonzero
     weight has a number D.
  4. (ii)  sum of D_e over the indexed edges of G''  <=  3 * sum of w_T over the listed triangles of G''
     (exact rational arithmetic).
  5. (i)   A = 3 M_w + 2 D is positive definite (floating point with the error bound above).
By the lemma in the paper, 2. to 5. imply that G'' is K4-free and that every 2-colouring of its edges has
a monochromatic triangle.

Usage:  python3 check_independent.py certificate.txt [--json out.json] [--no-matrix] [--sigma S] [--verbose]
Needs NumPy and SciPy.  Memory: one dense matrix of 8 m^2 bytes for m indexed edges (3.8 GB for the 483-vertex
graph, 6.3 GB for the 556-vertex graph; 1 GB = 10^9 bytes) and some more for everything else; on Linux and macOS
the program prints its peak memory at the end.  Run time: 2 to 4 minutes on 4 cores.
"""
import sys, time, json, math, argparse
from fractions import Fraction
import numpy as np

T0 = time.time()


def say(msg):
    print("[%7.1f s] %s" % (time.time() - T0, msg), flush=True)


def fail(msg):
    print("CHECK FAILED: " + msg, flush=True)
    sys.exit(1)


def need(cond, msg):
    if not cond:
        fail(msg)


# ------------------------------------------------------------------------------------------------------
#  reading the certificate
# ------------------------------------------------------------------------------------------------------
def read_certificate(path):
    cert = dict(removed=[], extra=[], classes=[], tri=[], dline=[], uniform=False)
    with open(path) as fh:
        for raw in fh:
            s = raw.split("#")[0].split()
            if not s:
                continue
            k = s[0]
            if k == "family":
                need(s[1:] == ["Q"], "family is not Q")
            elif k in ("p", "nr", "e0", "step", "orbits", "edge_orbits"):
                cert[k] = int(s[1])
            elif k == "theta":
                cert["theta"] = (int(s[1]), int(s[2]))
            elif k == "matrix":
                need(len(s) == 5 and "matrix" not in cert, "bad matrix line")
                cert["matrix"] = tuple(int(x) for x in s[1:])
            elif k in ("removed", "extra", "classes"):
                cert[k].extend(int(x) for x in s[1:])
            elif k == "uniform":
                cert["uniform"] = True
            elif k == "D":
                need(len(s) == 4, "bad D line")
                cert["dline"].append((int(s[1]), int(s[2]), Fraction(s[3])))
            else:
                need(len(s) == 4, "bad line: " + raw.strip()[:60])
                cert["tri"].append((int(s[0]), int(s[1]), int(s[2]), Fraction(s[3])))
    for k in ("p", "nr", "theta", "e0"):
        need(k in cert, "missing line " + k)
    need(("step" in cert) != ("matrix" in cert), "exactly one of the lines step and matrix is needed")
    need(cert["classes"] == [0], "this program handles only 'classes 0' (adjacency: det in H)")
    need(not (cert["uniform"] and (cert["tri"] or cert["dline"])), "uniform and weight lines together")
    return cert


# ------------------------------------------------------------------------------------------------------
#  the graph Q(p, e) from vectors
# ------------------------------------------------------------------------------------------------------
def build_q(p, e, nr, theta):
    need(p > 2 and all(p % r for r in range(2, int(math.isqrt(p)) + 1)), "p is not an odd prime")
    need((p - 1) % e == 0, "e does not divide p - 1")
    f = (p - 1) // e
    H = sorted(set(pow(x, e, p) for x in range(1, p)))
    need(len(H) == f, "the group of e-th powers has the wrong order")
    inH = np.zeros(p, dtype=bool)
    inH[H] = True
    need(bool(inH[p - 1]), "-1 is not in H")
    # sum-free: no a + b = c with a, b, c in H
    Ha = np.array(H, dtype=np.int64)
    sums = (Ha[:, None] + Ha[None, :]) % p
    sumfree = not bool(inH[sums].any())
    # classes of nonzero vectors modulo H: canonical member = the smallest code a*p + b in the class
    a = np.repeat(np.arange(p, dtype=np.int64), p)
    b = np.tile(np.arange(p, dtype=np.int64), p)
    best = np.full(p * p, p * p, dtype=np.int64)
    for h in H:
        np.minimum(best, ((a * h) % p) * p + (b * h) % p, out=best)
    canon = np.unique(best[1:])
    n = len(canon)
    need(n == (p + 1) * e, "number of classes is not (p+1)e")
    cls_of_code = np.searchsorted(canon, best)
    need(bool((np.bincount(cls_of_code[1:], minlength=n) == f).all()), "a class does not have |H| members")
    ra, rb = canon // p, canon % p
    det = (ra[:, None] * rb[None, :] - rb[:, None] * ra[None, :]) % p
    adjQ = inH[det]
    need(bool((adjQ == adjQ.T).all()) and not bool(adjQ.diagonal().any()), "adjacency of classes is not a graph")
    # the adjacency does not depend on the representatives (random test, the statement is det(hx, h'y) = hh' det(x, y))
    rng = np.random.default_rng(12345)
    c1 = rng.integers(1, p * p, size=2_000_000)
    c2 = rng.integers(1, p * p, size=2_000_000)
    d12 = ((c1 // p) * (c2 % p) - (c1 % p) * (c2 // p)) % p
    need(bool((inH[d12] == adjQ[cls_of_code[c1], cls_of_code[c2]]).all()), "adjacency depends on the representative")
    # translation of the labels: i -> class of theta^i
    ta, tb = theta[0] % p, theta[1] % p
    need(pow(nr % p, (p - 1) // 2, p) == p - 1, "nr is not a non-residue")
    lab = np.empty(n, dtype=np.int64)
    vec = np.empty((n, 2), dtype=np.int64)
    x0, x1 = 1, 0
    for i in range(n):
        vec[i] = (x0, x1)
        lab[i] = cls_of_code[x0 * p + x1]
        x0, x1 = (ta * x0 + nr * tb * x1) % p, (tb * x0 + ta * x1) % p
    need(len(set(lab.tolist())) == n, "i -> class of theta^i is not a bijection")
    adj = adjQ[np.ix_(lab, lab)]
    number_of_class = np.empty(n, dtype=np.int64)       # class -> the number i with class of theta^i = class
    number_of_class[lab] = np.arange(n)
    return dict(p=p, e=e, f=f, n=n, H=H, sumfree=sumfree, adj=adj, vec=vec, nr=nr % p,
                cls_of_code=cls_of_code, number_of_class=number_of_class)


# ------------------------------------------------------------------------------------------------------
#  main
# ------------------------------------------------------------------------------------------------------
def steps_1_to_4(args):
    out = {"certificate": args.certificate}

    cert = read_certificate(args.certificate)
    p, e = cert["p"], cert["e0"]
    Q = build_q(p, e, cert["nr"], cert["theta"])
    n, f, adj = Q["n"], Q["f"], Q["adj"]
    deg = adj.sum(axis=1)
    need(bool((deg == p).all()), "Q is not p-regular")
    cn = adj.astype(np.int32) @ adj.astype(np.int32)
    need(bool((cn[adj] == f).all()), "an edge of Q does not lie in exactly |H| triangles")
    say("1. Q(%d, %d): H = the %d-th powers mod %d, |H| = %d, -1 in H, H sum-free: %s." % (p, e, e, p, f, Q["sumfree"]))
    say("   %d classes of nonzero vectors modulo H; i -> class of theta^i (theta = %d + %d t, t^2 = %d) is a bijection;"
        % (n, cert["theta"][0], cert["theta"][1], Q["nr"]))
    say("   Q is regular of degree %d, with %d edges; every edge lies in exactly %d triangles."
        % (p, int(adj.sum()) // 2, f))
    out.update(p=p, e=e, H_order=f, sumfree=bool(Q["sumfree"]), n_Q=n)

    # ---- base graph G' and target graph G''
    removed = set(cert["removed"]); extra = set(cert["extra"])
    need(len(removed) == len(cert["removed"]) and len(extra) == len(cert["extra"]), "a vertex is listed twice")
    need(all(0 <= v < n for v in removed | extra) and not (removed & extra), "bad removed/extra lists")
    inbase = np.ones(n, dtype=bool); inbase[list(removed)] = False
    intarget = inbase.copy(); intarget[list(extra)] = False
    adjB = adj & inbase[:, None] & inbase[None, :]
    # description of the base vertex set by norms, when 3 divides p - 1 (information only)
    if removed and (p - 1) % 3 == 0:
        nrm = (Q["vec"][:, 0] ** 2 - Q["nr"] * Q["vec"][:, 1] ** 2) % p
        iscube = np.array([pow(int(x), (p - 1) // 3, p) == 1 for x in nrm])
        out["base_is_cube_norm_set"] = bool((iscube == inbase).all())
        if out["base_is_cube_norm_set"]:
            say("   information: the base vertex set is {[x]: x1^2 - %d x2^2 is a cube mod %d}; it is {i: i = 0 mod 3}: %s."
                % (Q["nr"], p, bool((inbase == (np.arange(n) % 3 == 0)).all())))
    tri = []                       # triangles of G', sorted triples
    tri_on_edge = {}
    nK4 = 0
    us, vs = np.nonzero(np.triu(adjB, 1))
    for u, v in zip(us.tolist(), vs.tolist()):
        c = np.flatnonzero(adjB[u] & adjB[v])
        if len(c) and adjB[np.ix_(c, c)].any():
            nK4 += 1
        tri_on_edge[(u, v)] = len(c)
        for w in c[c > v].tolist():
            tri.append((u, v, w))
    need(nK4 == 0, "the base graph contains a K4")
    triset = set(tri)
    edges = list(zip(us.tolist(), vs.tolist()))
    degB = sorted(set(adjB.sum(axis=1)[inbase].tolist()))
    tcount = sorted(set(tri_on_edge.values()))
    nvB = int(inbase.sum()); nvT = int(intarget.sum())
    eT = [(u, v) for (u, v) in edges if intarget[u] and intarget[v]]
    tT = [t for t in tri if intarget[t[0]] and intarget[t[1]] and intarget[t[2]]]
    say("2. base graph G': %d vertices, %d edges, %d triangles, degrees %s, triangles on an edge: %d to %d; no K4 (exhaustive)."
        % (nvB, len(edges), len(tri), degB if len(degB) > 1 else degB[0], tcount[0], tcount[-1]))
    say("   target graph G'' = G' minus %d vertices: %d vertices, %d edges, %d triangles (induced subgraph: no K4)."
        % (len(extra), nvT, len(eT), len(tT)))
    degT = adj[np.ix_(intarget, intarget)].sum(axis=1)
    out.update(base_vertices=nvB, base_edges=len(edges), base_triangles=len(tri), base_degrees=[int(x) for x in degB],
               tri_per_edge=[int(tcount[0]), int(tcount[-1])], target_vertices=nvT, target_edges=len(eT),
               target_triangles=len(tT), target_degrees=[int(degT.min()), int(degT.max())], extra=sorted(extra))

    # ---- the permutation pi of the vertex numbers that the certificate uses to shorten its lists
    if "matrix" in cert:
        ma, mb, mc, md = (x % p for x in cert["matrix"])
        y0 = (ma * Q["vec"][:, 0] + mc * Q["vec"][:, 1]) % p
        y1 = (mb * Q["vec"][:, 0] + md * Q["vec"][:, 1]) % p
        need(bool(((y0 != 0) | (y1 != 0)).all()), "the matrix maps a vertex to the zero vector")
        perm = Q["number_of_class"][Q["cls_of_code"][y0 * p + y1]]
        pi_text = "the map of the classes induced by (x1, x2) -> (%d x1 + %d x2, %d x1 + %d x2)" % (ma, mc, mb, md)
        out["symmetry"] = "matrix %d %d %d %d" % (ma, mb, mc, md)
    else:
        shift = (e * cert["step"]) % n
        need(shift != 0, "shift is zero")
        perm = (np.arange(n, dtype=np.int64) + shift) % n
        pi_text = "the shift i -> i + %d (mod %d)" % (shift, n)
        out["symmetry"] = "shift %d" % shift
    need(sorted(perm.tolist()) == list(range(n)), "pi is not a permutation of the vertices")
    # the cycles of pi; q = the least common multiple of their lengths
    cyc_len = np.zeros(n, dtype=np.int64)
    for v0 in range(n):
        if cyc_len[v0]:
            continue
        cyc = [v0]; x = int(perm[v0])
        while x != v0:
            cyc.append(x); x = int(perm[x])
        cyc_len[cyc] = len(cyc)
    q = 1
    for ln in set(cyc_len.tolist()):
        q = q * ln // math.gcd(q, ln)
    need(q >= 2, "pi is the identity")
    need(q <= 100000, "the order of pi is too large for this program")
    pl = perm.tolist()
    base_cycles = {}
    for v0 in np.flatnonzero(inbase).tolist():
        base_cycles[int(cyc_len[v0])] = base_cycles.get(int(cyc_len[v0]), 0) + 1
    cyc_text = ", ".join("%d of length %d" % (cnt // ln, ln) for ln, cnt in sorted(base_cycles.items(), reverse=True))
    w = {}                          # triangle -> Fraction
    d = {}                          # edge -> Fraction
    if cert["uniform"]:
        for t in tri:
            w[t] = Fraction(1)
        for ed in edges:
            d[ed] = Fraction(tri_on_edge[ed])
    else:
        for (a, b, c, wt) in cert["tri"]:
            need(0 <= min(a, b, c) and max(a, b, c) < n, "a vertex number is out of range")
            for k in range(q):
                t = tuple(sorted((a, b, c)))
                need(t in triset, "an image of a listed triple is not a triangle of G'")
                need(t not in w, "a triangle occurs twice")
                w[t] = wt
                a, b, c = pl[a], pl[b], pl[c]
        if cert["dline"]:
            edgeset = set(edges)
            for (u, v, dd) in cert["dline"]:
                need(0 <= min(u, v) and max(u, v) < n, "a vertex number is out of range")
                for k in range(q):
                    ed = (u, v) if u < v else (v, u)
                    need(ed in edgeset, "an image of a D-line is not an edge of G'")
                    need(ed not in d, "an edge occurs twice")
                    d[ed] = dd
                    u, v = pl[u], pl[v]
        else:
            for t, wt in w.items():
                if wt != 0:
                    for ed in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
                        d[ed] = d.get(ed, Fraction(0)) + wt
    for t, wt in w.items():
        if wt != 0:
            need((t[0], t[1]) in d and (t[0], t[2]) in d and (t[1], t[2]) in d, "an edge of a weighted triangle has no D")
    need(all(x > 0 for x in d.values()), "a number D is not positive")
    wmin, wmax = min(w.values()), max(w.values())
    say("3. certificate: pi = %s, a permutation of order %d;" % (pi_text, q))
    say("   the vertices of G' fall into the following cycles of pi: %s;" % cyc_text)
    say("   %d images of every line; %d triangles with a weight (%.6g to %.6g, %d of them negative, %d zero),"
        % (q, len(w), float(wmin), float(wmax), sum(1 for x in w.values() if x < 0), sum(1 for x in w.values() if x == 0)))
    say("   %d edges with a number D (%.6g to %.6g); all images are triangles or edges of G'; none occurs twice."
        % (len(d), float(min(d.values())), float(max(d.values()))))
    out.update(images=q, base_cycles={str(k): v for k, v in sorted(base_cycles.items())},
               weighted_triangles=len(w), indexed_edges=len(d),
               all_triangles_weighted=(len(w) == len(tri)), all_edges_indexed=(len(d) == len(edges)),
               w_range=[float(wmin), float(wmax)], d_range=[float(min(d.values())), float(max(d.values()))],
               negative_weights=int(sum(1 for x in w.values() if x < 0)))

    # ---- (ii), exact
    lhs = sum((dd for ed, dd in d.items() if intarget[ed[0]] and intarget[ed[1]]), Fraction(0))
    rhs = 3 * sum((wt for t, wt in w.items() if intarget[t[0]] and intarget[t[1]] and intarget[t[2]]), Fraction(0))
    nJ = sum(1 for ed in d if intarget[ed[0]] and intarget[ed[1]])
    need(nJ > 0, "no indexed edge in the target graph")
    budget_ok = lhs <= rhs
    say("4. (ii) exact: sum of D over the %d indexed edges of G'' = %s  %s  3 * sum of w over the triangles of G'' = %s"
        % (nJ, lhs, "<=" if budget_ok else ">", rhs))
    say("   (%d of the %d triangles of G'' have a weight in the certificate; the others count with the weight 0)"
        % (sum(1 for t in w if intarget[t[0]] and intarget[t[1]] and intarget[t[2]]), len(tT)))
    say("   as decimals: %.9f %s %.9f, ratio %.9f." % (float(lhs), "<=" if budget_ok else ">", float(rhs), float(lhs / rhs)))
    nTw = sum(1 for t in w if intarget[t[0]] and intarget[t[1]] and intarget[t[2]])
    out.update(budget_lhs=str(lhs), budget_rhs=str(rhs), budget_ratio=float(lhs / rhs), budget_ok=bool(budget_ok),
               indexed_edges_in_target=nJ, weighted_triangles_in_target=nTw)
    need(budget_ok, "the inequality (ii) fails")

    return out, d, w, nvT


# ------------------------------------------------------------------------------------------------------
#  step 5: positive definiteness of A without symmetry
# ------------------------------------------------------------------------------------------------------
def build_matrix(d, w):
    """N = S A S as lists of entries: (number of rows, rows, cols, vals of the off-diagonal part, diagonal)."""
    elist = sorted(d)
    m = len(elist)
    eidx = {ed: i for i, ed in enumerate(elist)}
    s = np.array([1.0 / math.sqrt(2.0 * float(d[ed])) for ed in elist])
    rows = []; cols = []; vals = []
    for t, wt in w.items():
        if wt == 0:
            continue
        i1, i2, i3 = eidx[(t[0], t[1])], eidx[(t[0], t[2])], eidx[(t[1], t[2])]
        x = 3.0 * float(wt)
        for (i, j) in ((i1, i2), (i1, i3), (i2, i3)):
            v = x * s[i] * s[j]
            rows += [i, j]; cols += [j, i]; vals += [v, v]
    rows = np.array(rows); cols = np.array(cols); vals = np.array(vals)
    need(len(np.unique(rows * m + cols)) == len(rows), "two edges lie in two listed triangles")
    diag = np.array([2.0 * float(d[ed]) for ed in elist]) * s * s
    return m, rows, cols, vals, diag


def cholesky_in_place(A, nb=256, progress=None):
    """Blocked left-looking Cholesky factorisation, written with matrix products only.  On return the lower
    triangle of A holds L and the part above the diagonal is zero.  Returns False if a diagonal block is
    not positive definite.  (How L is found does not matter for the proof: only the residual does.)"""
    from scipy.linalg import solve_triangular
    m = A.shape[0]
    for k in range(0, m, nb):
        k2 = min(m, k + nb)
        if k > 0:
            A[k:, k:k2] -= A[k:, :k] @ A[k:k2, :k].T
            A[:k, k:k2] = 0.0
        try:
            L11 = np.linalg.cholesky(A[k:k2, k:k2])
        except np.linalg.LinAlgError:
            return False
        A[k:k2, k:k2] = L11
        if k2 < m:
            A[k2:, k:k2] = solve_triangular(L11, A[k2:, k:k2].T, lower=True, check_finite=False).T
        if progress and (k // nb) % 10 == 0:
            progress("     factorisation: columns %d of %d" % (k2, m))
    return True


def residual_norm(Asp, L, chunk, progress=None):
    """Maximal absolute row sum of E = Asp - L L^T (computed), the largest |E_ij|, and the squared row norms of L."""
    m = L.shape[0]
    rowsum = np.zeros(m)
    ln2 = np.zeros(m)
    maxabs = 0.0
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        Lc = L[i0:i1, :i1]
        P = Lc @ L[:i1, :i1].T                           # rows i0..i1-1 of L L^T, columns 0..i1-1
        ln2[i0:i1] = np.einsum("ij,ij->i", Lc, Lc)
        E = Asp[i0:i1, :i1].toarray() - P
        E[:, i0:i1] = np.tril(E[:, i0:i1])               # keep the entries with column <= row only
        np.abs(E, out=E)
        maxabs = max(maxabs, float(E.max()))
        dg = E[np.arange(i1 - i0), np.arange(i0, i1)].copy()
        rowsum[i0:i1] += E.sum(axis=1)                   # entries (i, j), j <= i, counted for row i
        rowsum[:i1] += E.sum(axis=0)                     # the same entries counted for row j (symmetry)
        rowsum[i0:i1] -= dg                              # the diagonal entry was counted twice
        if progress and (i0 // chunk) % 10 == 0:
            progress("     residual: rows %d of %d" % (i1, m))
    return float(rowsum.max()), maxabs, ln2


def step_5(args, out, d, w, nvT):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh, LinearOperator
    from scipy.linalg import solve_triangular
    m, rows, cols, vals, diag = build_matrix(d, w)
    Nsp = (sp.csr_matrix((vals, (rows, cols)), shape=(m, m)) + sp.diags(diag)).tocsr()
    rowabs = np.asarray(abs(Nsp).sum(axis=1)).ravel()
    say("5. matrix N = S A S: %d rows, %d nonzero off-diagonal entries, diagonal 1, largest absolute row sum %.4f."
        % (m, len(vals), rowabs.max()))
    out.update(m=m, nnz_offdiag=int(len(vals)))

    def factor(sigma):
        A = np.zeros((m, m), order="F")
        A[rows, cols] = vals
        A[np.arange(m), np.arange(m)] = diag - sigma
        say("   Cholesky factorisation of N - sigma I, sigma = %.6e (dense, %d x %d, %.1f GB) ..."
            % (sigma, m, m, A.nbytes / 2 ** 30))
        if cholesky_in_place(A, progress=say if args.verbose else None):
            return A
        return None

    if args.sigma:
        sigma = args.sigma
    else:
        # A guess for the shift (not used in the proof): factor N itself, then estimate the smallest eigenvalue
        # of N by inverse iteration (Lanczos method for the inverse) with this factor.
        L = factor(0.0)
        need(L is not None, "the matrix N is not positive definite (the factorisation of N broke down)")

        def solve(x):
            y = solve_triangular(L, x, lower=True, check_finite=False)
            return solve_triangular(L, y, lower=True, trans="T", check_finite=False)
        op = LinearOperator((m, m), matvec=solve, dtype=np.float64)
        ev = eigsh(Nsp, k=1, sigma=0.0, which="LM", OPinv=op, tol=1e-4, ncv=20, return_eigenvectors=False)
        lam0 = float(ev[0])
        say("   estimate of the smallest eigenvalue of N (inverse iteration; a guess, not used in the proof): %.6e" % lam0)
        out["smallest_eigenvalue_estimate"] = lam0
        need(lam0 > 0, "the estimate of the smallest eigenvalue is not positive")
        del L, op
        sigma = 0.9 * lam0
    for attempt in range(5):
        L = factor(sigma)
        if L is not None:
            break
        say("   the factorisation broke down; trying a smaller sigma.")
        sigma /= 2
    else:
        fail("no Cholesky factorisation found")
    say("   factorisation done; computing the residual E = N - sigma I - L L^T entry by entry ...")
    Asp = (Nsp - sigma * sp.identity(m)).tocsr()
    normE, maxabs, ln2 = residual_norm(Asp, L, args.chunk, progress=say if args.verbose else None)
    u = 2.0 ** -53
    g = m * u / (1 - m * u)
    lmax2 = float(ln2.max()) * (1 + 1e-9)
    allow = g * m * lmax2 * 1.001 + 1e-12
    bound = sigma - normE * (1 + 1e-9) - allow
    say("   max |E_ij| = %.3e, ||E||_inf (computed) = %.3e, max |l_i|^2 = %.6f, allowance for rounding = %.3e"
        % (maxabs, normE, lmax2, allow))
    say("   PROVED:  lambda_min(N) >= sigma - ||E||_inf - allowance = %.6e" % bound)
    out.update(sigma=sigma, residual_max_entry=maxabs, residual_inf_norm=normE, lmax2=lmax2, rounding_allowance=allow,
               proved_lower_bound=bound, positive_definite=bool(bound > 0))
    need(bound > 0, "positive definiteness was not shown")
    try:                                                    # information only
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 1e9
        out["peak_memory_gb"] = peak
        say("   peak memory of this run: %.1f GB." % peak)
    except Exception:
        pass
    say("INDEPENDENT CHECK PASSED: the graph G'' has %d vertices and no K4, (ii) holds exactly, and A is positive" % nvT)
    say("definite (floating point with error bound, no symmetry used).  By the lemma, G'' arrows (3,3): F_e(3,3;4) <= %d." % nvT)
    out["passed"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate")
    ap.add_argument("--json")
    ap.add_argument("--no-matrix", action="store_true", help="skip step 5 (positive definiteness)")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="shift for step 5 (default: 0.9 times an estimate of the smallest eigenvalue)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out, d, w, nvT = steps_1_to_4(args)
    if args.no_matrix:
        say("step 5 skipped (--no-matrix).")
    else:
        step_5(args, out, d, w, nvT)
    out["seconds"] = time.time() - T0
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
