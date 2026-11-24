#!/usr/bin/env python3
"""
paper_graphs.py -- builds the graphs of the paper from the definitions that are printed in the paper and
compares each of them with the graph of its certificate file.  Not part of any proof: this is a check
that the text of the paper and the certificate files speak about the same graphs.

Usage:  python3 paper_graphs.py CERTIFICATE_DIRECTORY [--json out.json]

Definitions (p = 277, H = the sixth powers; ind = index to the base 5, a primitive root modulo 277;
[x] = the class of the nonzero vector x = (x1, x2) under multiplication by H):

  W     (483 vertices)  the classes [x] with x1 x2 != 0 and (ind(x1 x2) mod 12, ind(x1) mod 2) in
                        {(3,0), (4,0), (4,1), (5,1), (9,0), (10,1), (11,1)}
  W_489 (489 vertices)  the classes [x] with x1 x2 != 0 and (ind(x1 x2) mod 12, ind(x1) mod 2) in
                        {(2,1), (3,1), (4,0), (4,1), (5,0), (5,1), (8,0)}, and the six classes [(a, 0)]
  W_495 (495 vertices)  W_489 and the six classes [(0, a)]
  W_506 (506 vertices)  the classes [x] with x1 x2 != 0 and ind(x1 x2) mod 12 in {2, 3, 4, 5}, except those with
                        (ind x1 mod 12, ind x2 mod 12) in {(3,0), (9,6), (4,11), (10,5)}
  Y   (556 vertices)  the classes [x] for which x1^2 - 2 x2^2 is a cube modulo 277
  Y - X (534 and 533 vertices)  X is the set of the classes of theta^i for the numbers i in the line `extra`
                      of the certificate, theta = 1 + 3t, t^2 = 2
  Q(97,6) (588 vertices)  all classes (p = 97)

For every graph the program prints the numbers of vertices, edges and triangles, the degrees, the number of
triangles on an edge, whether the graph is connected, and the result of an exhaustive search for a K4.  If the
permutation pi of the certificate maps the graph to itself, it also prints the lengths of the orbits of pi on the
vertices, edges and triangles, and how many of the triangles with an orbit shorter than the order q of pi have
the form {v, pi^(q/3) v, pi^(2q/3) v}.
The routines of check_independent.py are used to build Q(p, e) and to translate the vertex numbers.
"""
import sys, os, json
import numpy as np
import check_independent as ci

P = 277
IND = {}
_x = 1
for _i in range(P - 1):
    IND[_x] = _i
    _x = _x * 5 % P
assert len(IND) == P - 1, "5 is not a primitive root modulo 277"

S_W = {(3, 0), (4, 0), (4, 1), (5, 1), (9, 0), (10, 1), (11, 1)}
S_W489 = {(2, 1), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (8, 0)}
Z_OUT = {(3, 0), (9, 6), (4, 11), (10, 5)}


def in_W(x1, x2):
    if x1 == 0 or x2 == 0:
        return False
    return ((IND[x1] + IND[x2]) % 12, IND[x1] % 2) in S_W


def in_W489(x1, x2):
    if x2 == 0:
        return True
    if x1 == 0:
        return False
    return ((IND[x1] + IND[x2]) % 12, IND[x1] % 2) in S_W489


def in_W495(x1, x2):
    return x1 == 0 or in_W489(x1, x2)


def in_Z(x1, x2):
    if x1 == 0 or x2 == 0:
        return False
    return (IND[x1] + IND[x2]) % 12 in (2, 3, 4, 5) and (IND[x1] % 12, IND[x2] % 12) not in Z_OUT


def in_Y(x1, x2):
    return pow((x1 * x1 - 2 * x2 * x2) % P, (P - 1) // 3, P) == 1


GRAPHS = [
    ("folkman_upper_483_q277s.txt", "W", in_W, False),
    ("folkman_upper_489_q277s.txt", "W_489", in_W489, False),
    ("folkman_upper_495_q277s.txt", "W_495", in_W495, False),
    ("folkman_upper_506_q277s.txt", "W_506", in_Z, False),
    ("folkman_upper_556_q277.txt", "Y", in_Y, False),
    ("folkman_upper_534_q277.txt", "Y - X (22 vertices removed)", in_Y, True),
    ("folkman_upper_533_q277.txt", "Y - X (23 vertices removed)", in_Y, True),
    ("folkman_upper_588_q97.txt", "Q(97,6)", None, False),
]


def facts(adj):
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    a32 = adj.astype(np.int32)
    cn = a32 @ a32
    us, vs = np.nonzero(np.triu(adj, 1))
    tri = int(cn[us, vs].sum()) // 3
    k4 = False
    for u, v in zip(us.tolist(), vs.tolist()):
        c = np.flatnonzero(adj[u] & adj[v])
        if len(c) > 1 and adj[np.ix_(c, c)].any():
            k4 = True
            break
    seen = np.zeros(n, dtype=bool); seen[0] = True; stack = [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u] & ~seen).tolist():
            seen[v] = True; stack.append(v)
    te = cn[us, vs]
    return dict(vertices=n, edges=len(us), triangles=tri, degree_min=int(deg.min()), degree_max=int(deg.max()),
                degrees=sorted(set(int(x) for x in deg)), tri_per_edge_min=int(te.min()), tri_per_edge_max=int(te.max()),
                connected=bool(seen.all()), K4=k4)


def permutation(cert, Q):
    """The permutation pi of the vertex numbers that the certificate names (line `step` or `matrix`)."""
    n = Q["n"]; p = cert["p"]
    if "matrix" in cert:
        ma, mb, mc, md = (x % p for x in cert["matrix"])
        y0 = (ma * Q["vec"][:, 0] + mc * Q["vec"][:, 1]) % p
        y1 = (mb * Q["vec"][:, 0] + md * Q["vec"][:, 1]) % p
        return Q["number_of_class"][Q["cls_of_code"][y0 * p + y1]]
    return (np.arange(n, dtype=np.int64) + cert["e0"] * cert["step"]) % n


def orbit_facts(adjfull, target, perm):
    """Orbit lengths of pi on the vertices, edges and triangles of the graph induced on `target`."""
    n = len(perm)
    q = 1; cur = perm.copy()
    while not (cur == np.arange(n)).all():
        cur = perm[cur]; q += 1
    adj = adjfull & target[:, None] & target[None, :]
    us, vs = np.nonzero(np.triu(adj, 1))
    edges = list(zip(us.tolist(), vs.tolist()))
    tris = []
    notri = 0
    for u, v in edges:
        c = np.flatnonzero(adj[u] & adj[v])
        notri += (len(c) == 0)
        for w in c[c > v].tolist():
            tris.append((u, v, w))

    def lengths(items):
        seen = set(); hist = {}; short = []
        for it in items:
            if it in seen:
                continue
            orb = []; cur = it
            while cur not in seen:
                seen.add(cur); orb.append(cur)
                cur = tuple(sorted(int(perm[x]) for x in cur))
            hist[len(orb)] = hist.get(len(orb), 0) + 1
            if len(orb) < q:
                short.extend(orb)
        return hist, short
    hv, _ = lengths([(v,) for v in np.flatnonzero(target).tolist()])
    he, _ = lengths(edges)
    ht, short = lengths(tris)
    rot = None
    if q % 3 == 0:
        p3 = np.arange(n)
        for _ in range(q // 3):
            p3 = perm[p3]
        rot = sum(1 for t in short if set(t) == {t[0], int(p3[t[0]]), int(p3[p3[t[0]]])} and int(p3[t[0]]) != t[0])
    return dict(order=q, vertex_orbits=hv, edge_orbits=he, triangle_orbits=ht, short_orbit_triangles=len(short),
                short_orbit_triangles_rotated=rot, edges_in_no_triangle=int(notri))


def main():
    cdir = sys.argv[1]
    out = {}
    ok = True
    cache = {}
    for fn, name, pred, minus_extra in GRAPHS:
        cert = ci.read_certificate(os.path.join(cdir, fn))
        key = (cert["p"], cert["e0"], cert["nr"], cert["theta"])
        if key not in cache:
            cache[key] = ci.build_q(*key)
        Q = cache[key]
        n = Q["n"]; vec = Q["vec"]
        target = np.ones(n, dtype=bool)
        target[cert["removed"]] = False
        target[cert["extra"]] = False
        if pred is None:
            defined = np.ones(n, dtype=bool)
        else:
            assert cert["p"] == P
            # the definition does not depend on the member of the class: test all members of every class
            Hs = Q["H"]
            for i in range(n):
                x1, x2 = int(vec[i][0]), int(vec[i][1])
                vals = set(pred(h * x1 % P, h * x2 % P) for h in Hs)
                assert len(vals) == 1, "the definition is not constant on a class"
            defined = np.array([pred(int(vec[i][0]), int(vec[i][1])) for i in range(n)])
            if minus_extra:
                assert all(defined[i] for i in cert["extra"]), "a vertex of X is not a vertex of Y"
                defined[cert["extra"]] = False
        same = bool((defined == target).all())
        ok = ok and same
        f = facts(Q["adj"][np.ix_(target, target)])
        f["definition_matches_certificate"] = same
        perm = permutation(cert, Q)
        invariant = bool((target[perm] == target).all())
        f["pi_maps_graph_to_itself"] = invariant
        if invariant:
            f["orbits"] = orbit_facts(Q["adj"], target, perm)
        out[fn] = f
        print("%-30s %-28s definition = certificate: %s" % (fn, name, same))
        print("     %d vertices, %d edges, %d triangles, degrees %d to %d, triangles on an edge %d to %d, connected: %s, K4: %s"
              % (f["vertices"], f["edges"], f["triangles"], f["degree_min"], f["degree_max"], f["tri_per_edge_min"],
                 f["tri_per_edge_max"], f["connected"], f["K4"]), flush=True)
        print("     degrees that occur: %s" % ", ".join(str(x) for x in f["degrees"]))
        if invariant:
            o = f["orbits"]
            fmt = lambda h: ", ".join("%d of length %d" % (h[k], k) for k in sorted(h, reverse=True))
            print("     pi has the order %d and maps the graph to itself; orbits on the vertices: %s;" % (o["order"], fmt(o["vertex_orbits"])))
            print("     on the edges: %s; on the triangles: %s;" % (fmt(o["edge_orbits"]), fmt(o["triangle_orbits"])))
            print("     triangles with a short orbit: %d%s; edges in no triangle: %d"
                  % (o["short_orbit_triangles"], "" if o["short_orbit_triangles_rotated"] is None else
                     " (%d of the form {v, pi^(q/3) v, pi^(2q/3) v})" % o["short_orbit_triangles_rotated"], o["edges_in_no_triangle"]), flush=True)
        else:
            print("     pi does not map the graph to itself (the certificate lives on a larger graph)", flush=True)
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump(out, fh, indent=1)
    print("ALL DEFINITIONS MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
