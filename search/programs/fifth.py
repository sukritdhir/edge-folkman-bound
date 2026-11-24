"""Which fifth coset?  Restrict given K'-symmetric weights of the 4-coset base to each 5-coset graph and evaluate.
usage: python3 exp/fifth.py weights.pkl [rm0]"""
import numpy as np, sys, pickle, time
sys.path.insert(0, ".")
from warm752 import build, tmask
from opt2 import evaluate

n = 941
src = sys.argv[1]; rm0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = pickle.load(open(src, "rb")); mu_o = np.array(d["mu"], float); D_o = np.array(d["D"], float)
G, So = build(n, 47, [0, 3, 7, 12], 0)
to = tmask(G, So)
orb_glob = -np.ones(len(G.tri), dtype=np.int64); orb_glob[np.nonzero(to)[0]] = So.tri_orbit
for c in range(20):
    if c in (0, 3, 7, 12): continue
    t0 = time.time()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        _, Sn = build(n, 47, [0, 3, 7, 12, c], rm0)
    tn = tmask(G, Sn)
    gi_new = np.nonzero(tn)[0]
    _, first = np.unique(Sn.tri_orbit, return_index=True)
    mu_n = mu_o[orb_glob[gi_new[first]]]
    D_n = np.zeros(Sn.nrep)
    ekn = np.nonzero(Sn.edge_rep >= 0)[0]
    D_n[Sn.edge_rep[ekn]] = D_o[So.edge_rep[ekn]]
    fac = 3 * (mu_n.sum() * Sn.q) / (D_n.sum() * Sn.q)
    D_n *= fac
    lam, _ = evaluate(Sn, mu_n, D_n, 2)
    print("fifth coset %2d: %d vertices, theta %.5f, lambda_min %.5f (margin %+.5f)  [%.0fs]" % (c, Sn.nv, fac, lam, lam + 2 / 3, time.time() - t0)); sys.stdout.flush()
