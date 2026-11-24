"""Interior-point (log-det barrier, Newton) solver for the K'-symmetric weighted spectral certificate (method A).

Graph G' = G_941 minus the K'-cosets `cosets` (minus vertex 0 if rm0 = 1); optional target G'' = G' - X.
Variables: mu (one per triangle orbit, sign free), delta (one per edge orbit),  delta = (2/3) D - Deg(mu):
    S_r(mu, delta) = U_r diag(mu) U_r^H + diag(delta)   positive semidefinite for all characters r of K'
(this is M + (2/3) D in the character basis).  Budget ratio
    beta = sum_a nf_a D_a / (3 sum_o mf_o mu_o) = 1 + c.y / 2      under the normalisation a.y = 1,
    c = (sum_p nf[oa[o,p]] - 2 mf_o ; nf_a),  a = (mf_o ; 0)        (nf, mf = surviving fractions; all 1 if X is empty).
A certificate exists iff min beta < 1; then lambda_min(N) + 2/3 = (2/3)(1 - beta) at the budget-tight scaling (about).
The barrier method follows the central path  t c + grad Phi(y) + nu a = 0,  Phi = -sum_r c_r log det S_r; on the path
beta(t) - min beta <= nubar / (2 t) with nubar = 47 * nrep, so the path gives two-sided bounds on the optimum.

usage: python3 exp/ipm_sym.py cosets rm0 init.pkl out.pkl [Xfile:m|none] [eps] [gap]
"""
import numpy as np, sys, os, time, pickle, ctypes
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.linalg.lapack import zpotrf, zpotri
sys.path.insert(0, "."); sys.path.insert(0, "exp")
from opt4 import setup, survival
from opt2 import evaluate

n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]; rm0 = int(sys.argv[2])
init = sys.argv[3]; outf = sys.argv[4]
xarg = sys.argv[5] if len(sys.argv) > 5 else "none"
eps = float(sys.argv[6]) if len(sys.argv) > 6 else 0.003
gap_target = float(sys.argv[7]) if len(sys.argv) > 7 else 3e-6
S_, G_, gen, h, R_, Sx = setup(n, 47, cosets, rm0)
q, nrep, no = Sx.q, Sx.nrep, Sx.no; NH = q // 2 + 1; m = no + nrep
nubar = float(q * nrep)
oa = np.ascontiguousarray(Sx.oa, dtype=np.int32); oi = np.ascontiguousarray(Sx.oi % q, dtype=np.int32)
cr = np.array([1.0] + [2.0] * (NH - 1))
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhess.so"))
lib.hess_sym.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 7 + [ctypes.c_int]
lib.hess_sym.restype = None

if xarg == "none":
    X = []; nf = np.ones(nrep); mf = np.ones(no)
else:
    f_, m_ = xarg.split(":"); X = [int(v) for v in pickle.load(open(f_, "rb"))[int(m_)][0]]
    nf, mf = survival(G_, Sx, X)
c_vec = np.concatenate([nf[oa].sum(axis=1) - 2 * mf, nf]); a_vec = np.concatenate([mf, np.zeros(nrep)])

cols = np.repeat(np.arange(no), 3)
U = []
for r in range(NH):
    psi = np.exp(2j * np.pi * r * oi / q)
    U.append(sp.csc_matrix((np.conj(psi).ravel(), (oa.ravel(), cols)), shape=(nrep, no)))
UH = [u.conj().T.tocsr() for u in U]
dg = np.diag_indices(nrep)


def blocks(mu, dl):
    out = np.empty((NH, nrep, nrep), dtype=complex)
    for r in range(NH):
        out[r] = (U[r] @ sp.diags(mu) @ UH[r]).toarray()
        out[r][dg] += dl
    return out


def factor(S):
    G = np.empty_like(S); ld = 0.0
    for r in range(NH):
        c, info = zpotrf(S[r], lower=1)
        if info != 0: return False, None, None
        ld += cr[r] * 2 * np.log(np.diag(c).real).sum()
        gi, info = zpotri(c, lower=1)
        gi = np.tril(gi); G[r] = gi + np.tril(gi, -1).conj().T
    return True, ld, G


def hessian(G):
    H = np.empty((m, m)); g = np.empty(m)
    lib.hess_sym(nrep, no, NH, q, G.ctypes.data, None, oa.ctypes.data, oi.ctypes.data, cr.ctypes.data, H.ctypes.data, g.ctypes.data, 0)
    return H, g


def linesearch(slope, e, w):
    """minimise  slope * alpha - sum_j w_j log(1 + alpha e_j)  over alpha > 0"""
    emin = e.min()
    hi = 0.999999 * (-1.0 / emin) if emin < 0 else 1e6
    def dphi(al): return slope - (w * e / (1 + al * e)).sum()
    if dphi(hi) <= 0: return hi
    lo = 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if dphi(mid) > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def save(y, tag):
    mu = y[:no]; dl = y[no:]
    D = 1.5 * (dl + Sx.degrees(mu))
    beta = (nf * D).sum() / (3 * (mf * mu).sum())
    Dt = D / beta
    lam, _ = evaluate(Sx, mu, Dt, 2)
    pickle.dump({"mu": mu, "D": Dt, "lam": lam, "beta": beta, "X": list(X), "cosets": cosets, "rm0": rm0}, open(outf, "wb"))
    print("   saved %s: beta = %.8f, lambda_min(N) at the tight budget = %.7f (margin %+.7f), min mu/mean = %.4f, min D/mean = %.4f %s"
          % (outf, beta, lam, lam + 2 / 3, mu.min() / mu.mean(), Dt.min() / Dt.mean(), tag)); sys.stdout.flush()
    return lam


if init == "uniform":
    mu = np.ones(no); D = Sx.degrees(mu)
else:
    d0 = pickle.load(open(init, "rb")); mu = np.array(d0["mu"], float)
    D = np.array(d0["D"], float) if d0.get("D") is not None else Sx.degrees(mu)
s = (mf * mu).sum(); mu = mu / s; D = D / s
D *= 3.0 / (nf * D).sum()
D *= (1 + eps)
y = np.concatenate([mu, (2.0 / 3.0) * D - Sx.degrees(mu)])
if os.environ.get("YSTART"): y = np.load(os.environ["YSTART"])
print("start: %d variables, nubar = %.0f, beta = %.6f" % (m, nubar, 1 + c_vec @ y / 2)); sys.stdout.flush()
S = blocks(y[:no], y[no:]); ok, ld, G = factor(S)
assert ok, "the starting point is not strictly feasible: increase eps"

if os.environ.get("FDTEST"):
    H, g = hessian(G)
    rng = np.random.default_rng(1); v = rng.standard_normal(m) * np.abs(y) * 1e-4
    S2 = blocks(y[:no] + v[:no], y[no:] + v[no:]); ok2, ld2, G2 = factor(S2); _, g2 = hessian(G2)
    print("FD test: |g2 - g - H v| / |H v| = %.2e;  (ld2 - ld) + (g + g2).v/2 = %.3e (ld change %.3e)"
          % (np.linalg.norm(g2 - g - H @ v) / np.linalg.norm(H @ v), -(ld2 - ld) - 0.5 * (g + g2) @ v, ld2 - ld)); sys.stdout.flush()

t = None; kappa = float(os.environ.get("KAPPA", "5")); it = 0; ncen = 0; t00 = time.time()
tol_c = float(os.environ.get("TOLC", "0.9")); FRAC = float(os.environ.get("FRAC", "0.9"))
while True:
    it += 1; t0 = time.time()
    H, gphi = hessian(G)
    t1 = time.time()
    sc = 1.0 / np.sqrt(np.diag(H))                                    # Jacobi scaling, then Cholesky (regularised if needed)
    H *= sc[:, None]; H *= sc[None, :]
    reg = 0.0
    while True:
        try:
            cf = cho_factor(H if reg == 0.0 else H + reg * np.eye(m), lower=True, check_finite=False); break
        except np.linalg.LinAlgError:
            reg = 1e-13 if reg == 0.0 else reg * 10
    def hsolve(b):
        x = cho_solve(cf, sc * b)
        if reg > 0:                                                   # two steps of iterative refinement with the true H
            for _ in range(2): x += cho_solve(cf, sc * b - H @ x)
        return sc * x
    v_c = hsolve(c_vec); v_a = hsolve(a_vec); v_g = hsolve(gphi)
    del H
    if t is None:
        A2 = np.array([[c_vec @ v_c, c_vec @ v_a], [c_vec @ v_a, a_vec @ v_a]]); b2 = -np.array([c_vec @ v_g, a_vec @ v_g])
        t = float(np.linalg.solve(A2, b2)[0])
        print("initial t = %.4g (gap bound nubar/2t = %.2e)" % (t, nubar / (2 * t))); sys.stdout.flush()
        if t <= 0: t = nubar / (2 * 0.01)
    def direction(t):
        nu = -(t * (a_vec @ v_c) + a_vec @ v_g) / (a_vec @ v_a)
        Dl = -(t * v_c + v_g + nu * v_a)
        return Dl, float(-(t * c_vec + gphi) @ Dl), nu
    Dl, lam2, nu = direction(t)
    msg = ""
    if lam2 < tol_c:
        beta = 1 + c_vec @ y / 2; gap = beta - (1 - nu / (2 * t))      # dual point Z = (G - G dS G)/t is feasible when lam2 < 1
        print("CENTRAL t = %.4g: beta = %.8f, lower bound on min beta about %.8f; margin now %+.7f, optimum at most %+.7f  (%d Newton steps, %.0fs total)"
              % (t, beta, beta - gap, (2 / 3) * (1 - beta), (2 / 3) * (1 - beta + gap), ncen, time.time() - t00)); sys.stdout.flush()
        save(y, "")
        if nubar / (2 * t) < gap_target: break
        if beta - gap > 1 + 2e-6 and not os.environ.get("NOSTOP"):
            print("STOP: the lower bound on beta exceeds 1: no certificate of this kind for this graph"); break
        if ncen <= 3: kappa = min(kappa * 1.5, 100.0)
        elif ncen > 7: kappa = max(kappa / 1.5, 1.5)
        t *= kappa; ncen = 0; msg = "t -> %.3g" % t
        Dl, lam2, nu = direction(t)
    dS = blocks(Dl[:no], Dl[no:])
    e = np.concatenate([eigh(dS[r], S[r], eigvals_only=True, check_finite=False) for r in range(NH)])
    w = np.repeat(cr, nrep)
    t2 = time.time()
    al = linesearch(t * (c_vec @ Dl), e, w)
    if e.min() < 0: al = min(al, FRAC * (-1.0 / e.min()))
    while True:
        S2 = S + al * dS; ok, ld2, G2 = factor(S2)
        if ok: break
        al *= 0.7
    y = y + al * Dl; S = S2; G = G2; ld = ld2; ncen += 1
    np.save(outf + ".y.npy", y)
    print("      diag: sum w e^2 / lam2 = %.4f, e min %.3f max %.3f, #|e|>0.3: %d, #|e|>0.1: %d, #|e|>0.03: %d; f_t = %.4f"
          % ((w * e * e).sum() / lam2, e.min(), e.max(), int((np.abs(e) > 0.3).sum()), int((np.abs(e) > 0.1).sum()), int((np.abs(e) > 0.03).sum()), t * (c_vec @ y) - ld))
    print("it %3d  t %.3g  beta %.8f  lam2 %.3e  alpha %.4f (max %.3f)  min mu/mean %.3f  [hess %.0fs, chol+eig %.0fs, fact %.0fs] reg %.0e %s"
          % (it, t, 1 + c_vec @ y / 2, lam2, al, (-1 / e.min() if e.min() < 0 else np.inf), y[:no].min() / y[:no].mean(), t1 - t0, t2 - t1, time.time() - t2, reg, msg)); sys.stdout.flush()
print("finished in %.0fs" % (time.time() - t00))
