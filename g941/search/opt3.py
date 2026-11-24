"""Proximal bundle optimisation of the weighted spectral certificate (triangle weights mu, free diagonal D)
on K'-symmetric induced subgraphs of a Lu-type circulant.  See opt2.py for the certificate statement.

Model: every eigenvector v of a block (normalised v^H D v = 1 at the point where it was computed) gives a cut
   c_v . z >= 0,   z = (mu, D),   c_v = ( v^H M_o v  for each triangle orbit o ,  (2/3) |v_a|^2 for each edge orbit a ).
Step: maximise  min_cuts (c . z) - |y - y_c|^2 / (2 tau)  over scaled variables y = z / z0 in the subspace
sum mu = const, sum D = 3 sum mu.  The dual is a small convex QP over the simplex, solved by FISTA.
"""
import numpy as np, sys, time, pickle
from symsub import SymSub, Sub, cyclic_subgroups
from opt2 import evaluate, cuts_from


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[k] - 1) / (k + 1.0), 0)


def simplex_qp(f, G, tau, theta0=None, iters=700):
    """min over the simplex of  f.theta + (tau/2) theta^T G theta."""
    n = len(f)
    th = np.full(n, 1.0 / n) if theta0 is None else theta0.copy()
    v = np.random.default_rng(0).normal(size=n)
    for _ in range(12):
        v = G @ v; v /= np.linalg.norm(v)
    L = 1.05 * tau * float(v @ (G @ v)) + 1e-12
    yv = th.copy(); tk = 1.0
    for _ in range(iters):
        g = f + tau * (G @ yv)
        new = proj_simplex(yv - g / L)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yv = new + ((tk - 1) / tn) * (new - th)
        th = new; tk = tn
    return th


def run(Sx, mu, D, iters=60, kk=24, tau=0.02, maxcuts=4200, tag="", save=None, stop_at=None):
    no, nr = Sx.no, Sx.nrep
    z0 = np.concatenate([mu, D])                       # fixed scaling
    B = np.zeros((2, no + nr)); B[0, :no] = mu; B[1, :no] = -3 * mu; B[1, no:] = D
    BBinv = np.linalg.inv(B @ B.T)
    def project_rows(C):
        return C - (C @ B.T) @ BBinv @ B
    yc = np.ones(no + nr)
    fc, eigs = evaluate(Sx, mu, D, kk)
    best = (fc, mu.copy(), D.copy())
    rm, rd = cuts_from(Sx, eigs)
    C = np.hstack([np.array(rm), np.array(rd)]) * z0[None, :]          # scaled cuts
    Cp = project_rows(C)
    G = Cp @ Cp.T
    print("%s start lambda_min = %.5f  (cuts %d)" % (tag, fc, C.shape[0])); sys.stdout.flush()
    theta = None
    for it in range(iters):
        t0 = time.time()
        f = C @ yc
        th0 = None
        if theta is not None and len(theta) <= len(f):
            th0 = np.zeros(len(f)); th0[:len(theta)] = theta
            if th0.sum() <= 0: th0 = None
        theta = simplex_qp(f, G, tau, th0)
        dlt = tau * (Cp.T @ theta)
        y = np.maximum(yc + dlt, 0.02)
        tmodel = float((C @ y).min())
        mut = y[:no] * z0[:no]; Dt = y[no:] * z0[no:]
        Dt = Dt * (3 * mut.sum() / Dt.sum())
        ft, eigs = evaluate(Sx, mut, Dt, kk)
        rm, rd = cuts_from(Sx, eigs)
        Cn = np.hstack([np.array(rm), np.array(rd)]) * z0[None, :]
        Cnp = project_rows(Cn)
        pred = tmodel - 2 / 3 - fc                       # predicted change of lambda_min
        act = ft - fc
        acc = act > 0.2 * max(pred, 0) and act > 0
        if acc:
            yc = y.copy(); fc = ft
            if ft > best[0]:
                best = (ft, mut.copy(), Dt.copy())
                if save: pickle.dump({"mu": best[1], "D": best[2], "lam": best[0]}, open(save, "wb"))
            if act > 0.6 * pred: tau *= 1.5
        else:
            tau *= 0.5
        # extend the bundle
        G = np.block([[G, Cp @ Cnp.T], [Cnp @ Cp.T, Cnp @ Cnp.T]])
        C = np.vstack([C, Cn]); Cp = np.vstack([Cp, Cnp])
        if C.shape[0] > maxcuts:                         # drop the cuts that are least active at the centre
            vals = C @ yc
            keep = np.sort(np.argsort(vals)[:maxcuts - 600])
            keep = np.union1d(keep, np.arange(C.shape[0] - Cn.shape[0], C.shape[0]))
            C = C[keep]; Cp = Cp[keep]; G = G[np.ix_(keep, keep)]; theta = None
        print("%s it %2d  trial %.5f  centre %.5f  best %.5f  predicted %+.5f actual %+.5f  tau %.4f  cuts %d  %s [%.0fs]"
              % (tag, it, ft, fc, best[0], pred, act, tau, C.shape[0], "accepted" if acc else "", time.time() - t0)); sys.stdout.flush()
        if tau < 1e-6: break
        if stop_at is not None and best[0] > stop_at: break
    return best


if __name__ == "__main__":
    n = 941
    S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
    G_ = Sub(n, S)
    def order(a):
        x = a; k = 1
        while x != 1: x = (x * a) % n; k += 1
        return k
    gen = next(a for a in range(2, n) if order(a) == n - 1)
    qk = int(sys.argv[1]); cosets = [int(c) for c in sys.argv[2].split(",")]
    rm0 = int(sys.argv[3]); iters = int(sys.argv[4])
    init = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "none" else None
    tau = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
    stop_at = float(sys.argv[7]) if len(sys.argv) > 7 else None
    h = pow(gen, (n - 1) // qk, n)
    Kp = [pow(h, t, n) for t in range(qk)]
    R = set((pow(gen, c, n) * s) % n for c in cosets for s in Kp) | ({0} if rm0 else set())
    Sx = SymSub(G_, h, R)
    mu = np.ones(Sx.no); D = None
    if init:
        d = pickle.load(open(init, "rb")); mu = np.array(d["mu"], dtype=float); D = np.array(d["D"]) if "D" in d else None
        print("initial weights from", init, "lam", d["lam"])
    mu = mu * (Sx.no / mu.sum())
    if D is None: D = Sx.degrees(mu)
    D = D * (3 * mu.sum() / D.sum())
    name = "K%d_%s_%d" % (qk, sys.argv[2].replace(",", "_"), rm0)
    best = run(Sx, mu, D, iters=iters, tau=tau, tag=name, save="scan/opt3_%s.pkl" % name, stop_at=stop_at)
    print("%s: n' = %d  best lambda_min = %.5f  (need > -0.66667)  %s" % (name, Sx.nv, best[0], "ARROWS" if best[0] > -2 / 3 else ""))
    pickle.dump({"n": n, "S": S, "h": h, "removed": sorted(R), "mu": best[1], "D": best[2], "lam": best[0], "oa": Sx.oa, "oi": Sx.oi},
                open("scan/opt3_final_%s.pkl" % name, "wb"))
