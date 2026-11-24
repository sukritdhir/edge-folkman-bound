"""Write the triangle system (m nt / e f g per line) of G_941 minus cosets (minus vertex 0 if rm0) for cpp/walk.
usage: python3 exp/mksys.py cosets rm0 out.txt"""
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
S_, G_, gen, h, R_, Sx = setup(941, 47, cosets, rm0)
ek = np.nonzero(Sx.edge_rep >= 0)[0]
idx = -np.ones(G_.E, dtype=np.int64); idx[ek] = np.arange(len(ek))
T = idx[Sx.tri_kept]
with open(sys.argv[3], "w") as f:
    f.write("%d %d\n" % (len(ek), len(T)))
    np.savetxt(f, T, fmt="%d")
print("wrote", sys.argv[3], len(ek), "edges", len(T), "triangles")
