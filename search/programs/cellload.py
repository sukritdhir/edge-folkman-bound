"""Per-orbit loads of a solver result: for every vertex orbit of the symmetry, the total weight of the triangles
that meet it and the total D of the edges that meet it (shares of the totals).
usage: python3 exp/cellload.py SPEC result.pkl"""
import sys, pickle
sys.path.insert(0, "exp")
import numpy as np, gsym

spec, pkl = sys.argv[1], sys.argv[2]
A, perm, removed, meta = gsym.from_spec(spec)
G_, Sx = gsym.make(A, perm, removed, drop_short=meta.get('drop_short', False))
d = pickle.load(open(pkl, "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float)
n = len(A); orb = gsym.split_orbits(np.asarray(perm))
tv = Sx.tri_verts[Sx.rep_tri]                       # representative triangles (global vertex numbers?)
ends = G_.ends[Sx.rep_edge]
print("spec %s: %d triangle orbits, %d edge orbits, q %d; beta %s" % (spec, Sx.no, Sx.nrep, Sx.q, d.get("beta")))
print("max vertex id in tri_verts:", int(tv.max()), " n base:", Sx.nv)
tri_orb = orb[tv]                                   # vertex orbits of the three corners of each representative
eo = orb[ends]
cells = sorted(set(int(x) for x in orb[[v for v in range(n) if v not in set(int(r) for r in removed)]]))
wtot = mu.sum(); dtot = D.sum()
rows = []
for c in cells:
    # all triangles meeting the cell: the orbit of a representative meets cell c iff one of its corners lies in c
    hit = (tri_orb == c)
    wl = (mu * hit.any(axis=1)).sum() / wtot           # share of triangle weight meeting the cell
    wc = (mu * hit.sum(axis=1)).sum() / (3 * wtot)     # share counted with multiplicity / 3 (sums to 1 over cells)
    dl = (D * (eo == c).sum(axis=1)).sum() / (2 * dtot)
    size = int((orb == c).sum())
    rows.append((wc / size, c, size, wl, wc, dl))
print("orbit  size  weight share (any corner)  weight share (corner count/3)  D share   per vertex (x n)")
nb = Sx.nv
for r in sorted(rows):
    print("%5d %5d   %.4f   %.4f   %.4f    %.3f" % (r[1], r[2], r[3], r[4], r[5], r[0] * nb))
