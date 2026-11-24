"""Write a compact text certificate (one representative triangle + decimal weight per orbit of <h>) from an
explicit .npz certificate with tied diagonal.  Weights are rounded to `digits` decimals (max weight = 1)."""
import numpy as np, sys
src, out = sys.argv[1], sys.argv[2]; digits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
title = sys.argv[4] if len(sys.argv) > 4 else ""
C = np.load(src)
assert "D" not in C.files, "text format supports the tied diagonal only"
n = int(C["n"]); h = int(C["h"]); tri = np.sort(np.array(C["tri"], dtype=np.int64), axis=1); w = np.array(C["w"], dtype=float)
w = w / w.max()
q = 1; x = h
while x != 1: x = (x * h) % n; q += 1
key = (tri[:, 0] * n + tri[:, 1]) * n + tri[:, 2]
order = np.argsort(key); skey = key[order]
seen = np.zeros(len(tri), dtype=bool); reps = []
for i in range(len(tri)):
    if seen[i]: continue
    t = tri[i].copy(); members = []
    for _ in range(q):
        k = np.sort(t); kk = (k[0] * n + k[1]) * n + k[2]
        j = order[np.searchsorted(skey, kk)]; assert key[j] == kk
        members.append(j); t = (t * h) % n
    assert len(set(members)) == q
    seen[members] = True
    wm = w[members]; assert np.abs(wm - wm[0]).max() < 1e-9
    reps.append((tuple(int(x) for x in tri[i]), round(float(wm[0]), digits)))
with open(out, "w") as f:
    f.write("# Weighted spectral certificate for an upper bound on the edge Folkman number F_e(3,3;4). %s\n" % title)
    f.write("# Graph: Cay(Z_n, S) minus the removed vertices.  Each data line 'a b c w' gives the weight w of the triangle\n")
    f.write("# {a,b,c} and of all its images {h^i a, h^i b, h^i c} mod n.  Verify with verify_spectral_cert.py.\n")
    f.write("n %d\nh %d\n" % (n, h))
    f.write("S " + " ".join(str(int(s)) for s in C["S"]) + "\n")
    f.write("removed " + " ".join(str(int(s)) for s in C["removed"]) + "\n")
    f.write("orbits %d\n" % len(reps))
    for (a, b, c), wt in reps: f.write("%d %d %d %.*f\n" % (a, b, c, digits, wt))
print("wrote", out, "orbits", len(reps), "min weight", min(r[1] for r in reps))
