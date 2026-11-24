"""Write a certificate for verify_q.py from a solver result of exp/ipm_g.py.
usage: python3 exp/export_q.py SPEC result.pkl certs/NAME.txt "title" [safety]
SPEC of the type Y (symmetry = shift, line 'step') or S (split torus, line 'matrix', needs verify_q2.py); the D values are scaled so that the budget ratio is 1 - safety
(default 1e-5 times ... see below) and all numbers are rounded to 12 significant digits (weights down, D up is not
needed: the verifier checks the budget inequality exactly on the printed numbers)."""
import sys, pickle
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np, gsym, qfam
from fractions import Fraction

spec, pkl, out, title = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
safety = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
A, perm, removed, meta = gsym.from_spec(spec)
G_, Sx = gsym.make(A, perm, removed, drop_short=meta.get('drop_short', False))
d = pickle.load(open(pkl, "rb"))
mu = np.array(d["mu"], float); D = np.array(d["D"], float); X = [int(x) for x in d.get("extra", [])]
assert len(mu) == Sx.no and len(D) == Sx.nrep
nf, mf = gsym.survival(G_, Sx, X)
D = D * (3 * (mf * mu).sum() / (nf * D).sum()) * (1 - safety)          # budget ratio 1 - safety
sc = 8.0 / D.max(); mu = mu * sc; D = D * sc                           # normalisation: max D = 8
# fixed-point decimals with 12 places: the common denominator of the file divides 10^12 (the verifier needs
# 2 * D_e * denominator < 2^48)
fmt = lambda x: "%.12f" % x
ws = [fmt(x) for x in mu]; ds = [fmt(x) for x in D]
# exact budget of the printed numbers
lhs = sum(Fraction(s) * int(round(c * Sx.q)) for s, c in zip(ds, nf)); rhs = 3 * sum(Fraction(s) * int(round(c * Sx.q)) for s, c in zip(ws, mf))
assert lhs <= rhs, "rounding broke the budget; increase safety"
f = spec.split(":")
p, e0 = meta["p"], meta["e0"]
nr = next(a for a in range(2, p) if pow(a, (p - 1) // 2, p) == p - 1)
with open(out, "w") as fh:
    fh.write("# %s\n" % title)
    fh.write("# spec %s; solver result %s; budget ratio of the printed numbers %.9f\n" % (spec, pkl, float(lhs / rhs)))
    fh.write("# format: see verify_q.py.  Lines 'a b c w': weight w of the triangle {a,b,c} and of its images under pi;\n")
    fh.write("# lines 'D u v d': the number D for the edge {u,v} and its images under pi.\n")
    fh.write("family Q\np %d\nnr %d\ntheta %d %d\ne0 %d\nclasses 0\n" % (p, nr, meta["theta"][0], meta["theta"][1], e0))
    if "matrix" in meta: fh.write("matrix %d %d %d %d\n" % tuple(meta["matrix"]))        # needs verify_q2.py
    else: fh.write("step %d\n" % meta["step"])
    rem = sorted(int(x) for x in removed)
    for i in range(0, len(rem), 40): fh.write("removed " + " ".join(str(x) for x in rem[i:i + 40]) + "\n")
    if X: fh.write("extra " + " ".join(str(x) for x in sorted(X)) + "\n")
    fh.write("orbits %d\nedge_orbits %d\n" % (Sx.no, Sx.nrep))
    for o in range(Sx.no):
        a, b, c = (int(x) for x in Sx.tri_verts[Sx.rep_tri[o]])
        fh.write("%d %d %d %s\n" % (a, b, c, ws[o]))
    for a in range(Sx.nrep):
        u, v = (int(x) for x in G_.ends[Sx.rep_edge[a]])
        fh.write("D %d %d %s\n" % (u, v, ds[a]))
print("wrote %s: %d vertices in the base, %d extra, %d triangle orbits, %d edge orbits, budget ratio %.9f" % (out, Sx.nv, len(X), Sx.no, Sx.nrep, float(lhs / rhs)))
