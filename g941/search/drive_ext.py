"""Staged proximal-bundle driver for extension certificates (see opt4.py for the statement).
Optimise (mu, D) for the current extra vertex set X; whenever the margin exceeds `target`, record the stage,
add the cheapest further vertex to X and continue WITH THE SAME BUNDLE (the cuts v^H (M + 2/3 D) v >= 0 do not
depend on the budget; only the two linear constraints change).
usage: python3 drive_ext.py cosets Xfile:m init.pkl minutes target_margin [tau0]
Stages: scan/drive_m{m}.pkl (mu, D, lam, X) and scan/driveX.pkl ({m: (X, None, lam)}; export_ext.py reads 'scan/driveX.pkl:m').
Current state: scan/drive_cur.pkl (mu, D, lam, X)."""
import numpy as np, sys, time, pickle, os
from opt4 import setup, survival
from opt2 import evaluate, cuts_from
from opt3 import simplex_qp

n = 941
cosets = [int(c) for c in sys.argv[1].split(",")]
f, m = sys.argv[2].split(":"); X = [int(v) for v in pickle.load(open(f, "rb"))[int(m)][0]]
d = pickle.load(open(sys.argv[3], "rb")); mu = np.array(d["mu"], float); D = np.array(d["D"], float)
deadline = time.time() + 60 * float(sys.argv[4]); target = float(sys.argv[5])
tau_main = float(sys.argv[6]) if len(sys.argv) > 6 else 300.0
WARM = 6; tau = 2.0                                     # warm-up: collect cuts near the start with tiny steps
BUNDLE = "scan/drive_bundle.npy"                         # raw (unscaled) cuts; they do not depend on X
S, G_, gen, h, R, Sx = setup(n, 47, cosets)
ek = np.nonzero(Sx.edge_rep >= 0)[0]
tri = Sx.tri_kept
tv = np.sort(np.concatenate([G_.ends[tri[:, 0]], G_.ends[tri[:, 1]], G_.ends[tri[:, 2]]], axis=1), axis=1)[:, [0, 2, 4]]
ebase = np.zeros(G_.E, dtype=bool); ebase[ek] = True
stages = pickle.load(open("scan/driveX.pkl", "rb")) if os.path.exists("scan/driveX.pkl") else {}
no, nr = Sx.no, Sx.nrep; kk = 24; maxcuts = 4200


def next_vertex(mu, D, X):
    De = np.zeros(G_.E); De[ek] = D[Sx.edge_rep[ek]]; wt = mu[Sx.tri_orbit]
    bad = np.zeros(n, dtype=bool); bad[list(X)] = True
    at = ~bad[tv].any(axis=1); ae = ebase & ~bad[G_.ends[:, 0]] & ~bad[G_.ends[:, 1]]
    W = wt[at].sum(); Ds = De[ae].sum()
    wl = np.zeros(n); dl = np.zeros(n)
    for j in range(3): np.add.at(wl, tv[at, j], wt[at])
    aei = np.nonzero(ae)[0]
    for j in range(2): np.add.at(dl, G_.ends[aei, j], De[aei])
    th = 3 * (W - wl) / (Ds - dl); th[~Sx.keep] = -1; th[bad] = -1
    return int(np.argmax(th))


nf, mf = survival(G_, Sx, X)
D = D * (3 * (mf * mu).sum() / (nf * D).sum())
z0 = np.concatenate([mu, D])
def set_budget(nf, mf):
    B = np.zeros((2, no + nr)); B[0, :no] = z0[:no]; B[1, :no] = -3 * mf * z0[:no]; B[1, no:] = nf * z0[no:]
    return B, np.linalg.inv(B @ B.T)
B, BBinv = set_budget(nf, mf)
def project_rows(C):
    return C - (C @ B.T) @ BBinv @ B
yc = np.ones(no + nr)
fc, eigs = evaluate(Sx, mu, D, kk)
rm, rd = cuts_from(Sx, eigs)
C = np.hstack([np.array(rm), np.array(rd)]) * z0[None, :]
if os.path.exists(BUNDLE):
    raw = np.load(BUNDLE)
    if raw.shape[1] == no + nr:
        C = np.vstack([raw * z0[None, :], C]); WARM = 0; tau = tau_main
        print("loaded %d cuts from %s" % (raw.shape[0], BUNDLE)); sys.stdout.flush()
    del raw
Cp = project_rows(C); G = Cp @ Cp.T
print("start: |X| = %d, lambda_min = %.6f (margin %+.6f)" % (len(X), fc, fc + 2 / 3)); sys.stdout.flush()
theta = None; it = 0
while time.time() < deadline:
    it += 1; t0 = time.time()
    fvec = C @ yc
    th0 = None
    if theta is not None and len(theta) <= len(fvec):
        th0 = np.zeros(len(fvec)); th0[:len(theta)] = theta
        if th0.sum() <= 0: th0 = None
    theta = simplex_qp(fvec, G, tau, th0)
    y = np.maximum(yc + tau * (Cp.T @ theta), 0.02)
    tmodel = float((C @ y).min())
    mut = y[:no] * z0[:no]; Dt = y[no:] * z0[no:]
    fac = 3 * (mf * mut).sum() / (nf * Dt).sum(); Dt = Dt * fac; y[no:] *= fac
    ft, eigs = evaluate(Sx, mut, Dt, kk)
    rm, rd = cuts_from(Sx, eigs)
    Cn = np.hstack([np.array(rm), np.array(rd)]) * z0[None, :]
    Cnp = project_rows(Cn)
    pred = tmodel - 2 / 3 - fc; act = ft - fc
    acc = act > 0.2 * max(pred, 0) and act > 0
    if acc:
        yc = y.copy(); fc = ft
        pickle.dump({"mu": mut, "D": Dt, "lam": fc, "X": list(X)}, open("scan/drive_cur.pkl", "wb"))
    if it <= WARM:
        if it == WARM: tau = tau_main
    elif acc:
        if pred < 5e-6: tau *= 3.0                           # model sees no gain: the step is far too short
        elif act > 0.6 * pred: tau *= 1.5
    else:
        tau *= 0.5
    if os.path.exists("scan/drive_tau.txt"):                 # manual override
        try: tau = float(open("scan/drive_tau.txt").read().split()[0])
        except Exception: pass
        os.remove("scan/drive_tau.txt")
    if it % 10 == 0: np.save(BUNDLE, C / z0[None, :])
    G = np.block([[G, Cp @ Cnp.T], [Cnp @ Cp.T, Cnp @ Cnp.T]])
    C = np.vstack([C, Cn]); Cp = np.vstack([Cp, Cnp])
    if C.shape[0] > maxcuts:
        vals = C @ yc
        keep = np.sort(np.argsort(vals)[:maxcuts - 600])
        keep = np.union1d(keep, np.arange(C.shape[0] - Cn.shape[0], C.shape[0]))
        C = C[keep]; Cp = Cp[keep]; G = G[np.ix_(keep, keep)]; theta = None
    print("m %2d it %3d  trial %.5f  centre %.5f  predicted %+.5f actual %+.5f  tau %.2f  cuts %d %s [%.0fs]"
          % (len(X), it, ft, fc, pred, act, tau, C.shape[0], "accepted" if acc else "", time.time() - t0)); sys.stdout.flush()
    if tau < 1e-3: tau = 1e-3
    if fc > -2 / 3 + target:                                   # stage finished: record it and add a vertex
        muc = yc[:no] * z0[:no]; Dc = yc[no:] * z0[no:]
        pickle.dump({"mu": muc, "D": Dc, "lam": fc, "X": list(X)}, open("scan/drive_m%d.pkl" % len(X), "wb"))
        stages[len(X)] = (list(X), None, fc); pickle.dump(stages, open("scan/driveX.pkl", "wb"))
        print("STAGE DONE: %d extra vertices -> %d vertices, lambda %.6f (margin %+.6f)" % (len(X), Sx.nv - len(X), fc, fc + 2 / 3)); sys.stdout.flush()
        X = X + [next_vertex(muc, Dc, X)]
        nf, mf = survival(G_, Sx, X)
        fac = 3 * (mf * muc).sum() / (nf * Dc).sum(); yc[no:] *= fac
        B, BBinv = set_budget(nf, mf)
        Cp = project_rows(C); G = Cp @ Cp.T; theta = None
        fc, eigs = evaluate(Sx, yc[:no] * z0[:no], yc[no:] * z0[no:], kk)
        print("new stage: |X| = %d (added vertex %d), lambda_min = %.6f (margin %+.6f)" % (len(X), X[-1], fc, fc + 2 / 3)); sys.stdout.flush()
print("driver finished; stages:", sorted(stages))
