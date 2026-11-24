"""Proximal bundle optimisation of an EXTENSION certificate.

Base graph G' = G_941 minus some K'-cosets (K'-symmetric).  Target graph G'' = G' - X for an arbitrary vertex set X.
Variables: K'-symmetric triangle weights mu (per triangle orbit of G') and a K'-symmetric positive diagonal D
(per edge orbit of G').  If
     (i)  M(mu) + (2/3) D  is positive definite on the edge space of G'      (47 small Hermitian blocks), and
     (ii) sum_{e in E(G'')} D_e  <=  3 * sum_{T triangle of G''} mu_T        (budget counted on G'' only),
then G'' arrows (3,3): a colouring x of E(G'') without monochromatic triangle, extended by zero to E(G'), has
x^T (M + (2/3) D) x = -2 W'' + (2/3) sum_{E''} D_e <= 0, contradicting (i).
(The principal submatrix of M on E(G'') only sees triangles of G''.)

So the only change against opt3.py is the budget: orbit a counts with n_a = number of its edges avoiding X and
orbit o with m_o = number of its triangles avoiding X.
"""
import numpy as np, sys, time, pickle
from symsub import SymSub, Sub, cyclic_subgroups
from opt2 import evaluate, cuts_from
from opt3 import simplex_qp


def survival(G, Sx, X):
    """n_a / q and m_o / q for the vertex set X (list of vertices of G')."""
    bad = np.zeros(G.n, dtype=bool); bad[list(X)] = True
    ek = Sx.edge_rep >= 0
    eok = ek & ~bad[G.ends[:, 0]] & ~bad[G.ends[:, 1]]
    na = np.bincount(Sx.edge_rep[eok], minlength=Sx.nrep).astype(float)
    tri = Sx.tri_kept
    tok = eok[tri[:, 0]] & eok[tri[:, 1]] & eok[tri[:, 2]]
    mo = np.bincount(Sx.tri_orbit[tok], minlength=Sx.no).astype(float)
    return na / Sx.q, mo / Sx.q


def run(Sx, mu, D, nf, mf, iters=60, kk=24, tau=0.02, maxcuts=4200, tag="", save=None, stop_at=None, tau_min=1e-6, deadline=None):
    no, nr = Sx.no, Sx.nrep
    D = D * (3 * (mf * mu).sum() / (nf * D).sum())
    z0 = np.concatenate([mu, D])                       # fixed scaling
    B = np.zeros((2, no + nr)); B[0, :no] = mu; B[1, :no] = -3 * mf * mu; B[1, no:] = nf * D
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
    if save: pickle.dump({"mu": best[1], "D": best[2], "lam": best[0]}, open(save, "wb"))
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
        Dt = Dt * (3 * (mf * mut).sum() / (nf * Dt).sum())
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
        G = np.block([[G, Cp @ Cnp.T], [Cnp @ Cp.T, Cnp @ Cnp.T]])
        C = np.vstack([C, Cn]); Cp = np.vstack([Cp, Cnp])
        if C.shape[0] > maxcuts:                         # drop the cuts that are least active at the centre
            vals = C @ yc
            keep = np.sort(np.argsort(vals)[:maxcuts - 600])
            keep = np.union1d(keep, np.arange(C.shape[0] - Cn.shape[0], C.shape[0]))
            C = C[keep]; Cp = Cp[keep]; G = G[np.ix_(keep, keep)]; theta = None
        print("%s it %2d  trial %.5f  centre %.5f  best %.5f  predicted %+.5f actual %+.5f  tau %.4f  cuts %d  %s [%.0fs]"
              % (tag, it, ft, fc, best[0], pred, act, tau, C.shape[0], "accepted" if acc else "", time.time() - t0)); sys.stdout.flush()
        if tau < tau_min: break
        if stop_at is not None and best[0] > stop_at: break
        if it >= 8 and pred < 2e-6 and acc: break        # converged
        if deadline is not None and time.time() > deadline: break
    return best


def setup(n, qk, cosets, rm0=None):
    """rm0: also remove the fixed vertex 0 (default: environment variable RM0, else no)."""
    import os
    if rm0 is None: rm0 = int(os.environ.get("RM0", "0"))
    S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
    G_ = Sub(n, S)
    def order(a):
        x = a; k = 1
        while x != 1: x = (x * a) % n; k += 1
        return k
    gen = next(a for a in range(2, n) if order(a) == n - 1)
    h = pow(gen, (n - 1) // qk, n)
    Kp = [pow(h, t, n) for t in range(qk)]
    R = set((pow(gen, c, n) * s) % n for c in cosets for s in Kp) | ({0} if rm0 else set())
    return S, G_, gen, h, R, SymSub(G_, h, R)


if __name__ == "__main__":
    # python3 opt4.py cosets partial_coset m_list iters init_pkl tau [rm0]
    n = 941; qk = 47
    cosets = [int(c) for c in sys.argv[1].split(",")]
    xfile = sys.argv[2] if sys.argv[2].endswith(".pkl") else None       # greedy X lists (greedy_x.py) or a coset index
    pc = None if xfile else int(sys.argv[2]); ms = [int(x) for x in sys.argv[3].split(",")]
    iters = int(sys.argv[4]); init = sys.argv[5]; tau = float(sys.argv[6])
    rm0 = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    S, G_, gen, h, R, Sx = setup(n, qk, cosets)
    d = pickle.load(open(init, "rb")); mu = np.array(d["mu"], dtype=float); D = np.array(d["D"], dtype=float)
    assert len(mu) == Sx.no and len(D) == Sx.nrep
    for m in ms:
        if xfile:
            X = [int(v) for v in pickle.load(open(xfile, "rb"))[m][0]]
            name = "ext_%s_g_m%d" % (sys.argv[1].replace(",", "_"), m)
        else:
            X = [(pow(gen, pc, n) * pow(h, t, n)) % n for t in range(m)] + ([0] if rm0 else [])
            name = "ext_%s_p%d_m%d_%d" % (sys.argv[1].replace(",", "_"), pc, m, rm0)
        nf, mf = survival(G_, Sx, X)
        best = run(Sx, mu, D, nf, mf, iters=iters, tau=tau, tag=name, save="scan/%s.pkl" % name)
        mu, D = best[1], best[2]
        nv = Sx.nv - len(X)
        print("%s: n'' = %d  best lambda_min = %.5f  (need > -0.66667)  %s" % (name, nv, best[0], "ARROWS" if best[0] > -2 / 3 else "")); sys.stdout.flush()
        pickle.dump({"n": n, "S": S, "h": h, "removed": sorted(R), "extra": sorted(X), "mu": best[1], "D": best[2], "lam": best[0],
                     "oa": Sx.oa, "oi": Sx.oi, "nf": nf, "mf": mf}, open("scan/%s_final.pkl" % name, "wb"))
