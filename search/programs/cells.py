"""Cell coordinates of the orbits of the order-23 split torus element of Q(p, e0): the orbit of [(alpha, beta)] is
determined by (ind alpha mod 2 e0, ind beta mod 2 e0) up to adding (e0, e0).  Prints the grid of a set of orbits.
usage: python3 exp/cells.py p e0 order O:list | J:classes"""
import sys
sys.path.insert(0, "exp")
import numpy as np, qfam, gsym, splitclass

def cell_table(Q, orb):
    p = Q.p; e0 = Q.e0; g = Q.g; c = 2 * e0
    ind = np.full(p, -1, dtype=np.int64); y = 1
    for i in range(p - 1): ind[y] = i; y = (y * g) % p
    cells = {}
    for v in range(Q.n):
        x0, x1 = int(Q.X[v, 0]), int(Q.X[v, 1])
        if x0 == 0 or x1 == 0: continue
        a, b = int(ind[x0]) % c, int(ind[x1]) % c
        if a >= e0: a -= e0; b = (b - e0) % c
        o = int(orb[v])
        assert cells.get(o, (a, b)) == (a, b)
        cells[o] = (a, b)
    return cells

if __name__ == "__main__":
    p, e0, order = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    Q = qfam.QGraph(p, e0); perm = gsym.split_perm(Q, order); orb = gsym.split_orbits(perm)
    cells = cell_table(Q, orb); c = 2 * e0
    for spec in sys.argv[4:]:
        kind, arg = spec.split(":")
        if kind == "J":
            cl = splitclass.split_classes(Q); sel = sorted(set(int(o) for o in orb[np.isin(cl, [int(x) for x in arg.split(",")])]))
        else: sel = [int(x) for x in arg.split(",")]
        grid = [["." for _ in range(c)] for _ in range(e0)]
        for o in sel:
            a, b = cells[o]; grid[a][b] = "#"
        print(spec[:60], "(%d orbits)   rows: a = ind(alpha) mod %d (0..%d), columns: b = ind(beta) mod %d" % (len(sel), c, e0 - 1, c))
        for a in range(e0): print("   a=%d  " % a + " ".join(grid[a]) + "     a+b mod 12: " + ",".join(str((a + b) % c) for b in range(c) if grid[a][b] == "#"))
