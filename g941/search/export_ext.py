"""Write a compact text EXTENSION certificate: K'-symmetric triangle weights and diagonal on the base graph
(G_941 minus cosets), plus the list `extra` of further removed vertices.  Verify with certs/verify_spectral_cert.py.
usage: python3 export_ext.py cosets weights.pkl extra(comma list | greedy.pkl:m | none) out.txt [title]"""
import numpy as np, sys, pickle
from opt4 import setup, survival
from opt2 import evaluate

n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]
d = pickle.load(open(sys.argv[2], "rb"))
xs = sys.argv[3]
if xs == "none": X = []
elif ".pkl:" in xs:
    f, m = xs.split(":"); X = [int(v) for v in pickle.load(open(f, "rb"))[int(m)][0]]
elif xs == "pkl": X = [int(v) for v in d["extra"]]
else: X = [int(v) for v in xs.split(",")]
out = sys.argv[4]; title = sys.argv[5] if len(sys.argv) > 5 else ""
S, G, gen, h, R, Sx = setup(n, 47, cosets)
mu = np.array(d["mu"], float); D = np.array(d["D"], float) if "D" in d and d["D"] is not None else Sx.degrees(mu)
assert len(mu) == Sx.no and len(D) == Sx.nrep
if "oa" in d: assert (Sx.oa == d["oa"]).all() and (Sx.oi == d["oi"]).all(), "orbit numbering mismatch"
sc = mu.max(); mu = mu / sc; D = D / sc
mur = np.array([float("%.6f" % v) for v in mu])
assert (mur > 0).all()
nf, mf = survival(G, Sx, X)
D = D * ((1 - 1e-5) * 3 * (mf * mur).sum() / (nf * D).sum())
Dr = np.array([float("%.6g" % v) for v in D])
ratio = (nf * Dr).sum() / (3 * (mf * mur).sum())
assert ratio <= 1 - 1e-6, ratio
lam, _ = evaluate(Sx, mur, Dr, 2)
print("rounded certificate: budget ratio %.9f, lambda_min %.6f, margin %+.6f" % (ratio, lam, lam + 2 / 3))
_, first = np.unique(Sx.tri_orbit, return_index=True)
tv = np.sort(G.ends[Sx.tri_kept[first]].reshape(len(first), 6), axis=1)[:, [0, 2, 4]]
erep = {}
for e in np.nonzero(Sx.edge_rep >= 0)[0]:
    erep.setdefault(int(Sx.edge_rep[e]), e)
with open(out, "w") as f:
    f.write("# Weighted spectral certificate for an upper bound on the edge Folkman number F_e(3,3;4). %s\n" % title)
    f.write("# Base graph: Cay(Z_n, S) minus the vertices `removed`.  Target graph: the base graph minus the vertices `extra`.\n")
    f.write("# Each line 'a b c w' gives the weight w of the triangle {a,b,c} and of all its images under v -> h^i v mod n.\n")
    f.write("# Each line 'D u v d' gives the diagonal entry d of the edge {u,v} and of all its images.\n")
    f.write("# Verify with verify_spectral_cert.py (it explains the criterion and checks everything from scratch).\n")
    f.write("n %d\nh %d\n" % (n, h))
    f.write("S " + " ".join(str(int(s)) for s in sorted(S)) + "\n")
    f.write("removed " + " ".join(str(int(s)) for s in sorted(R)) + "\n")
    if X: f.write("extra " + " ".join(str(int(s)) for s in sorted(X)) + "\n")
    f.write("orbits %d\nedge_orbits %d\n" % (Sx.no, Sx.nrep))
    for (a, b, c), wt in zip(tv, mur): f.write("%d %d %d %.6f\n" % (a, b, c, wt))
    for a in range(Sx.nrep):
        u, v = G.ends[erep[a]]; f.write("D %d %d %.6g\n" % (u, v, Dr[a]))
print("wrote", out, ": base", Sx.nv, "vertices, extra", len(X), "-> target", Sx.nv - len(X), "vertices")
