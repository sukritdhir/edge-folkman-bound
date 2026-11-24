#!/usr/bin/env python3
"""
export_graphs.py -- writes the graphs of the paper as plain files, for readers who want to test them with
their own tools.  Not part of any proof.

Usage:  python3 export_graphs.py CERTIFICATE OUTPREFIX [--base]

For the graph G of Lemma 3 of the paper, which is the graph that arrows (3,3) (it is called G'' in the output
of verify_q2.py), or, with --base, for the graph G' on which the certificate is defined, the program writes
  OUTPREFIX.edges.txt     first line "n m", then one edge "u v" per line, 0 <= u < v < n
  OUTPREFIX.g6            the same graph in the graph6 format (one line)
  OUTPREFIX.vertices.txt  for each vertex u: its number i in the certificate and the vector theta^i = (a, b),
                          so that the vertex is the class of (a, b) in Q(p, e)
The graph is built by the routines of check_independent.py (from the definition of Q(p, e) on vectors).
"""
import sys
import numpy as np
import check_independent as ci


def graph6(adj):
    n = adj.shape[0]
    if n <= 62:
        head = [n + 63]
    else:
        head = [126, ((n >> 12) & 63) + 63, ((n >> 6) & 63) + 63, (n & 63) + 63]
    bits = []
    for j in range(1, n):
        bits.extend(adj[:j, j].astype(np.uint8).tolist())
    while len(bits) % 6:
        bits.append(0)
    b = np.array(bits, dtype=np.uint8).reshape(-1, 6)
    vals = (b * np.array([32, 16, 8, 4, 2, 1], dtype=np.uint8)).sum(axis=1) + 63
    return bytes(head + vals.tolist()).decode("ascii")


def main():
    cert_path, prefix = sys.argv[1], sys.argv[2]
    base = "--base" in sys.argv[3:]
    cert = ci.read_certificate(cert_path)
    Q = ci.build_q(cert["p"], cert["e0"], cert["nr"], cert["theta"])
    n = Q["n"]
    keep = np.ones(n, dtype=bool)
    keep[cert["removed"]] = False
    if not base:
        keep[cert["extra"]] = False
    idx = np.flatnonzero(keep)
    adj = Q["adj"][np.ix_(idx, idx)]
    us, vs = np.nonzero(np.triu(adj, 1))
    with open(prefix + ".edges.txt", "w") as fh:
        fh.write("%d %d\n" % (len(idx), len(us)))
        for u, v in zip(us.tolist(), vs.tolist()):
            fh.write("%d %d\n" % (u, v))
    with open(prefix + ".g6", "w") as fh:
        fh.write(graph6(adj) + "\n")
    with open(prefix + ".vertices.txt", "w") as fh:
        fh.write("# vertex   number i in the certificate   theta^i = (a, b)   [Q(%d, %d), theta = %d + %d t, t^2 = %d]\n"
                 % (cert["p"], cert["e0"], cert["theta"][0], cert["theta"][1], Q["nr"]))
        for u, i in enumerate(idx.tolist()):
            fh.write("%d %d %d %d\n" % (u, i, Q["vec"][i][0], Q["vec"][i][1]))
    print("%s: %d vertices, %d edges" % (prefix, len(idx), len(us)))


if __name__ == "__main__":
    main()
