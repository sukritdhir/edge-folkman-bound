import sys, pickle, time, numpy as np
sys.path.insert(0, "exp")
from sublib import *
d = pickle.load(open(sys.argv[1], "rb")); mu = np.array(d["mu"]); D = np.array(d["D"]); X = list(d["X"])
T = Target(X)
D = D * (3 * (T.mf * mu).sum() / (T.nf * D).sum())
print("|X| = %d, core orbits %d, fringe edges %d, |F| = %d" % (len(X), T.nco, T.nfr, len(T.F)))
N2, dme = sparse_N2(T, mu, D)
t0 = time.time(); lam0 = lanczos_min(N2, T.ein, 1500); print("Lanczos %.6f [%.0fs]" % (lam0, time.time() - t0))
rng = np.random.default_rng(5); V = rng.standard_normal((NE, 128))
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, lam0 - GAP0, 4)
print("theta[0..3]", th[:4], "res", res[0], res.max(), "times", tim)
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, th[0] - GAP0, 2)
print("theta[0..3]", th[:4], "res", res[0], res.max(), "times", tim)
z = np.concatenate([mu, D]); val = rm @ mu + rd @ D
print("cut values vs theta + 2/3: max dev %.2e" % np.abs(val - th - 2 / 3).max())
t0 = time.time(); print("inertia count at margin/2:", certify(T, mu, D, 0.5 * (th[0] + 2 / 3)), "[%.0fs]" % (time.time() - t0))
print("inertia count just above the bottom:", certify(T, mu, D, th[0] + 2 / 3 + 3e-5))
