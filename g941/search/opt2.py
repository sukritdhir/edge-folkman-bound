"""Trust-region cutting-plane optimisation of the weighted spectral certificate on K'-symmetric subgraphs,
with a free diagonal.

Certificate: triangle weights mu_T and a positive diagonal D on the edges with  sum_e D_e = 3 sum_T mu_T  and
     M(mu) + (2/3) D  positive definite     (equivalently lambda_min(D^(-1/2) M D^(-1/2)) > -2/3).
Then the graph arrows (3,3): a colouring x without monochromatic triangle has x^T M x = -2 mu(T) and
x^T D x = sum_e D_e, so x^T (M + (2/3) D) x = 0.            (D = Deg(mu) is the special case used before.)
"""
import numpy as np, sys, time, pickle
from scipy.optimize import linprog
from symsub import SymSub, Sub, cyclic_subgroups


def evaluate(Sx, mu, D, kk):
    eigs = {r: Sx.eig(mu, r, kk, D) for r in range(Sx.q // 2 + 1)}
    return min(v[0][0] for v in eigs.values()), eigs


def cuts_from(Sx, eigs, c=2.0 / 3.0):
    rm, rd = [], []
    for r, (lam, U, idx, dm) in eigs.items():
        w = np.exp(2j * np.pi * r / Sx.q)
        for i in range(len(lam)):
            v = np.zeros(Sx.nrep, dtype=complex); v[idx] = U[:, i] * dm
            b = np.zeros(Sx.no)
            for (p, s) in Sx.pairs:
                b += 2 * np.real(np.conj(v[Sx.oa[:, p]]) * v[Sx.oa[:, s]] * w ** ((Sx.oi[:, s] - Sx.oi[:, p]) % Sx.q))
            rm.append(b); rd.append(c * np.abs(v) ** 2)
    return rm, rd


def run(Sx, mu, D, iters=40, kk=10, rho=1.15, maxcuts=5000, tag="", save=None, free_diag=True, stop_at=None):
    no, nr = Sx.no, Sx.nrep
    fbest, eigs = evaluate(Sx, mu, D, kk)
    best = (fbest, mu.copy(), D.copy())
    RM, RD = cuts_from(Sx, eigs)
    print("%s start  lambda_min = %.5f" % (tag, fbest)); sys.stdout.flush()
    for it in range(iters):
        t0 = time.time()
        if len(RM) > maxcuts: RM = RM[-maxcuts:]; RD = RD[-maxcuts:]
        A = np.hstack([np.array(RM), np.array(RD)])
        nc = A.shape[0]
        mub, Db = best[1], best[2]
        cvec = np.zeros(no + nr + 1); cvec[-1] = -1.0
        A_ub = np.hstack([-A, np.ones((nc, 1))]); b_ub = np.zeros(nc)
        A_eq = np.zeros((2, no + nr + 1)); A_eq[0, :no] = 1.0; A_eq[1, :no] = -3.0; A_eq[1, no:no + nr] = 1.0
        b_eq = [mub.sum(), 0.0]
        if free_diag:
            bounds = [(mub[i] / rho, mub[i] * rho) for i in range(no)] + [(Db[i] / rho, Db[i] * rho) for i in range(nr)] + [(None, None)]
            res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.status != 0: print("LP failed", res.message); break
            mut = res.x[:no]; Dt = res.x[no:no + nr]; tmodel = -res.fun
        else:                                           # tied diagonal D = Deg(mu): substitute
            Mdeg = np.zeros((nr, no))
            for p in range(3): np.add.at(Mdeg, (Sx.oa[:, p], np.arange(no)), 1.0)
            A2 = A[:, :no] + A[:, no:] @ Mdeg
            c2 = np.zeros(no + 1); c2[-1] = -1.0
            res = linprog(c2, A_ub=np.hstack([-A2, np.ones((nc, 1))]), b_ub=b_ub, A_eq=np.hstack([np.ones((1, no)), np.zeros((1, 1))]), b_eq=[mub.sum()],
                          bounds=[(mub[i] / rho, mub[i] * rho) for i in range(no)] + [(None, None)], method="highs")
            if res.status != 0: print("LP failed", res.message); break
            mut = res.x[:no]; Dt = Sx.degrees(mut); tmodel = -res.fun
        ft, eigs = evaluate(Sx, mut, Dt, kk)
        rm, rd = cuts_from(Sx, eigs); RM += rm; RD += rd
        acc = ft > best[0] + 1e-6
        if acc:
            best = (ft, mut.copy(), Dt.copy()); rho = min(1.0 + (rho - 1.0) * 1.3, 2.0)
            if save: pickle.dump({"mu": best[1], "D": best[2], "lam": best[0]}, open(save, "wb"))
        else:
            rho = 1.0 + (rho - 1.0) * 0.5
        print("%s it %2d  trial %.5f  best %.5f  model %.5f  cuts %d  rho %.3f  %s [%.0fs]" % (tag, it, ft, best[0], tmodel - 2 / 3, nc, rho, "accepted" if acc else "", time.time() - t0)); sys.stdout.flush()
        if rho < 1.0005: break
        if stop_at is not None and best[0] > stop_at: break
    return best


if __name__ == "__main__":
    n = 941
    S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
    G = Sub(n, S)
    def order(a):
        x = a; k = 1
        while x != 1: x = (x * a) % n; k += 1
        return k
    gen = next(a for a in range(2, n) if order(a) == n - 1)
    qk = int(sys.argv[1]); cosets = [int(c) for c in sys.argv[2].split(",")]
    rm0 = int(sys.argv[3]); iters = int(sys.argv[4])
    init = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "none" else None
    free_diag = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    h = pow(gen, (n - 1) // qk, n)
    Kp = [pow(h, t, n) for t in range(qk)]
    R = set((pow(gen, c, n) * s) % n for c in cosets for s in Kp) | ({0} if rm0 else set())
    Sx = SymSub(G, h, R)
    mu = np.ones(Sx.no); D = None
    if init:
        d = pickle.load(open(init, "rb")); mu = np.array(d["mu"], dtype=float); D = np.array(d["D"]) if "D" in d else None
        print("initial weights from", init, "lam", d["lam"])
    mu = mu * (Sx.no / mu.sum())
    if D is None: D = Sx.degrees(mu)
    D = D * (3 * mu.sum() / D.sum())
    name = "K%d_%s_%d" % (qk, sys.argv[2].replace(",", "_"), rm0)
    best = run(Sx, mu, D, iters=iters, tag=name, save="scan/opt2_%s.pkl" % name, free_diag=bool(free_diag))
    print("%s: n' = %d  best lambda_min = %.5f  (need > -0.66667)  %s" % (name, Sx.nv, best[0], "ARROWS" if best[0] > -2 / 3 else ""))
    pickle.dump({"n": n, "S": S, "h": h, "removed": sorted(R), "mu": best[1], "D": best[2], "lam": best[0], "oa": Sx.oa, "oi": Sx.oi},
                open("scan/opt2_final_%s.pkl" % name, "wb"))
