"""Staged proximal-bundle optimiser for METHOD B certificates.

Base graph G' = G_941 minus K'-cosets (K' = <302>, order 47), optionally minus vertex 0; target graph G'' = G' - X.
Variables: K'-symmetric triangle weights mu (one per triangle orbit of G') and a K'-symmetric diagonal D (one per
edge orbit).  Objective: lambda_min of N'' = the principal submatrix of N = D^(-1/2) M D^(-1/2) on the edges of
G'', with the budget  sum_{E''} D_e = 3 * sum_{T in G''} mu_T.   (N'' > -2/3 certifies that G'' arrows (3,3).)

Eigenvectors of N'' come from a shift-invert subspace iteration (exp/sublib.py): the edge orbits that lie completely
in G'' form a K'-symmetric core (block-diagonal over the characters of K'), the remaining edges of G'' (the fringe:
edges at the kept vertices of partly deleted cosets) are handled by a Schur complement.
Every Ritz vector u (supported on E'') gives a valid cut  x^T (M'' + 2/3 D'') x >= 0  with x = D^(-1/2) u.

usage: [RM0=1] python3 exp/drive_sub.py cosets Xfile:m init.pkl minutes target_margin [tau_main] [k]
(RM0=1: vertex 0 is removed from the base graph as well; Xfile 'none' starts with the cheapest single vertex)
Stages: scan/{TAG}_m{m}.pkl (mu, D, lam, X), scan/{TAG}X.pkl ({m: (X, None, lam)}); current state scan/{TAG}_cur.pkl;
TAG = environment variable SUBTAG (default "sub").  Step size override while running: echo 100 > scan/{TAG}_tau.txt
"""
import sys, os
os.environ["COSETS"] = sys.argv[1]
sys.path.insert(0, "exp")
from sublib import *

if sys.argv[2] == "none": X = []
else:
    f_, m_ = sys.argv[2].split(":"); X = [int(v) for v in pickle.load(open(f_, "rb"))[int(m_)][0]]
d0 = pickle.load(open(sys.argv[3], "rb")); mu = np.array(d0["mu"], float); D = np.array(d0["D"], float)
deadline = time.time() + 60 * float(sys.argv[4]); target = float(sys.argv[5])
tau_main = float(sys.argv[6]) if len(sys.argv) > 6 else 300.0
KV = int(sys.argv[7]) if len(sys.argv) > 7 else 128
TAG = os.environ.get("SUBTAG", "sub")

# ------------------------------------------------------------------ main loop
if not X: X = [next_vertex(mu, D, X)]
T = Target(X)
D = D * (3 * (T.mf * mu).sum() / (T.nf * D).sum())
# variable scaling: an orbit of which only a fraction f survives in the target enters every cut with weight f, so its
# variable is rescaled by 1/sqrt(f) (diagonal preconditioning of the proximal step); dead orbits keep scale 1
sv = np.concatenate([np.where(T.mf > 0, 1 / np.sqrt(np.maximum(T.mf, 1.0 / q)), 1.0), np.where(T.nf > 0, 1 / np.sqrt(np.maximum(T.nf, 1.0 / q)), 1.0)])
if os.environ.get("NOSCALE"): sv[:] = 1.0
z0 = np.concatenate([mu, D]) * sv
def set_budget(T):
    B = np.zeros((2, no + nrep)); B[0, :no] = z0[:no]; B[1, :no] = -3 * T.mf * z0[:no]; B[1, no:] = T.nf * z0[no:]
    return B, np.linalg.inv(B @ B.T)
B, BBinv = set_budget(T)
def project_rows(C):
    return C - (C @ B.T) @ BBinv @ B
stages = pickle.load(open("scan/%sX.pkl" % TAG, "rb")) if os.path.exists("scan/%sX.pkl" % TAG) else {}
yc = 1.0 / sv
N2, _ = sparse_N2(T, mu, D)
lam0 = lanczos_min(N2, T.ein, 1500)
print("start: |X| = %d, |F| = %d, Lanczos estimate of lambda_min(N'') = %.6f (margin %+.6f)" % (len(X), len(T.F), lam0, lam0 + 2 / 3)); sys.stdout.flush()
rng = np.random.default_rng(5)
V = rng.standard_normal((NE, KV))
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, lam0 - GAP0, 4)
th, V, res, rm, rd, tim = evaluate(T, mu, D, V, th[0] - GAP0, 2)
fc = float(th[0])
print("subspace: theta[0..3] = %s, theta[k-1] = %.6f, residuals %.1e .. %.1e, times %.0f/%.0f/%.0fs, max|inv| %.1e"
      % (np.array2string(th[:4], precision=6), th[-1], res[0], res.max(), tim[0], tim[1], tim[2], tim[3])); sys.stdout.flush()
C = np.hstack([rm, rd]) * z0[None, :]
Cp = project_rows(C); Gm = Cp @ Cp.T
Vhist = [V.astype(np.float32)]; NHIST = 14                # recent Ritz vectors, to rebuild valid cuts after a stage change
tau = 2.0; WARM = 4; maxcuts = 4200
theta = None; it = 0
while time.time() < deadline:
    it += 1; t0 = time.time()
    fvec = C @ yc
    th0 = None
    if theta is not None and len(theta) <= len(fvec):
        th0 = np.zeros(len(fvec)); th0[:len(theta)] = theta
        if th0.sum() <= 0: th0 = None
    theta = simplex_qp(fvec, Gm, tau, th0)
    y = np.maximum(yc + tau * (Cp.T @ theta), 0.02 / sv)
    tmodel = float((C @ y).min())
    mut = y[:no] * z0[:no]; Dt = y[no:] * z0[no:]
    fac = 3 * (T.mf * mut).sum() / (T.nf * Dt).sum(); Dt = Dt * fac; y[no:] *= fac
    tq = time.time() - t0
    th, Vt, res, rm, rd, tim = evaluate(T, mut, Dt, V, fc - GAP0, 2)
    ft = float(th[0])
    if ft < fc - 0.6 * GAP0:                                  # the bottom moved close to (or below) the shift: redo
        th, Vt, res, rm, rd, tim = evaluate(T, mut, Dt, Vt, ft - GAP0, 2); ft = float(th[0])
    V = Vt
    Vhist.append(V.astype(np.float32)); Vhist = Vhist[-NHIST:]
    Cn = np.hstack([rm, rd]) * z0[None, :]
    Cnp = project_rows(Cn)
    pred = tmodel - 2 / 3 - fc; act = ft - fc
    ratio = act / max(pred, 1e-12)
    acc = act > 0 and ratio > 0.05
    if acc:
        yc = y.copy(); fc = ft
        pickle.dump({"mu": mut, "D": Dt, "lam": fc, "X": list(X)}, open("scan/%s_cur.pkl" % TAG, "wb"))
    if it <= WARM:
        if it == WARM: tau = tau_main
    elif acc:
        if pred < 5e-6: tau *= 3.0
        elif ratio > 0.6: tau *= 1.5
        elif ratio < 0.2: tau *= 0.6
    else:
        tau *= 0.5
    if os.path.exists("scan/%s_tau.txt" % TAG):
        try: tau = float(open("scan/%s_tau.txt" % TAG).read().split()[0])
        except Exception: pass
        os.remove("scan/%s_tau.txt" % TAG)
    Gm = np.block([[Gm, Cp @ Cnp.T], [Cnp @ Cp.T, Cnp @ Cnp.T]])
    C = np.vstack([C, Cn]); Cp = np.vstack([Cp, Cnp])
    if C.shape[0] > maxcuts:
        vals = C @ yc
        keep = np.sort(np.argsort(vals)[:maxcuts - 600])
        keep = np.union1d(keep, np.arange(C.shape[0] - Cn.shape[0], C.shape[0]))
        C = C[keep]; Cp = Cp[keep]; Gm = Gm[np.ix_(keep, keep)]; theta = None
    print("m %2d it %3d  trial %.6f  centre %.6f  pred %+.6f act %+.6f  tau %.1f  cuts %d %s res %.0e/%.0e inv %.0e [qp %.0fs, blocks %.0fs, eig %.0fs, cuts %.0fs]"
          % (len(X), it, ft, fc, pred, act, tau, C.shape[0], "acc" if acc else "   ", res[0], res.max(), tim[3], tq, tim[0], tim[1], tim[2])); sys.stdout.flush()
    if tau < 1e-3: tau = 1e-3
    if fc > -2 / 3 + target:
        muc = yc[:no] * z0[:no]; Dc = yc[no:] * z0[no:]
        cnt, nneg = certify(T, muc, Dc, 0.5 * target)
        print("inertia check at -2/3 + %.6f: %d eigenvalues of N'' below (the base matrix has %d)" % (0.5 * target, cnt, nneg)); sys.stdout.flush()
        if cnt == 0:
            pickle.dump({"mu": muc, "D": Dc, "lam": fc, "X": list(X)}, open("scan/%s_m%d.pkl" % (TAG, len(X)), "wb"))
            stages[len(X)] = (list(X), None, fc); pickle.dump(stages, open("scan/%sX.pkl" % TAG, "wb"))
            print("STAGE DONE: %d extra vertices -> %d vertices, lambda_min(N'') %.6f (margin %+.6f)" % (len(X), Sx.nv - len(X), fc, fc + 2 / 3)); sys.stdout.flush()
            X = X + [next_vertex(muc, Dc, X)]
            T = Target(X)
            fac = 3 * (T.mf * muc).sum() / (T.nf * Dc).sum(); yc[no:] *= fac
            B, BBinv = set_budget(T)
            mun = yc[:no] * z0[:no]; Dn = yc[no:] * z0[no:]
            N2, dme = sparse_N2(T, mun, Dn)
            lam0 = lanczos_min(N2, T.ein, 1200)
            th, V, res, rm, rd, tim = evaluate(T, mun, Dn, V, lam0 - GAP0, 3)
            fc = float(th[0])
            # old cuts belong to vectors that live partly on the deleted edges: rebuild cuts from the recent Ritz vectors
            rows_m = [rm]; rows_d = [rd]
            for Vh in Vhist:
                Vz = Vh.astype(float) * T.ein[:, None]
                Vz /= np.linalg.norm(Vz, axis=0)[None, :]
                a, b = cuts(T, Vz, dme); rows_m.append(a); rows_d.append(b)
            Vhist = [(Vh * T.ein[:, None]).astype(np.float32) for Vh in Vhist] + [V.astype(np.float32)]; Vhist = Vhist[-NHIST:]
            C = np.hstack([np.vstack(rows_m), np.vstack(rows_d)]) * z0[None, :]
            Cp = project_rows(C); Gm = Cp @ Cp.T; theta = None
            print("new stage: |X| = %d (added vertex %d), |F| = %d, lambda_min(N'') = %.6f (margin %+.6f)" % (len(X), X[-1], len(T.F), fc, fc + 2 / 3)); sys.stdout.flush()
        else:
            print("WARNING: Ritz value above the true minimum; refreshing the subspace"); sys.stdout.flush()
            th, V, res, rm, rd, tim = evaluate(T, muc, Dc, V, fc - 2 * GAP0, 4)
            fc = float(th[0])
print("driver finished; stages:", sorted(stages))
