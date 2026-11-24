"""Unit triangle weights, free diagonal: how far does the *unweighted* SDP (MAX-CUT relaxation of the triangle
graph, as in Lange-Radziszowski-Xu) get on the 753-vertex graph?  Cutting-plane optimisation over D only.
Reuses the research thread's block code (symsub.py, opt2.py).  Output: best lambda_min(D^-1/2 M D^-1/2), needs > -2/3."""
import sys, time, pickle
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, ".")
from symsub import SymSub, Sub, cyclic_subgroups
from opt2 import evaluate, cuts_from

n = 941
S = [s for s in cyclic_subgroups(n) if len(s) == 188][0]
G = Sub(n, S)
def order(a):
    x = a; k = 1
    while x != 1: x = (x * a) % n; k += 1
    return k
gen = next(a for a in range(2, n) if order(a) == n - 1)
cosets = [int(c) for c in sys.argv[1].split(",")]
iters = int(sys.argv[2]); out = sys.argv[3]
h = pow(gen, (n - 1) // 47, n)
Kp = [pow(h, t, n) for t in range(47)]
R = set((pow(gen, c, n) * s) % n for c in cosets for s in Kp)
Sx = SymSub(G, h, R)
no, nr = Sx.no, Sx.nrep
mu = np.ones(no)
D = Sx.degrees(mu); D = D * (3 * mu.sum() / D.sum())
kk = 12
fbest, eigs = evaluate(Sx, mu, D, kk)
best = (fbest, D.copy())
RM, RD = cuts_from(Sx, eigs)
print("start (D = triangle degrees): lambda_min = %.5f" % fbest, flush=True)
rho = 1.15
for it in range(iters):
    t0 = time.time()
    if len(RM) > 6000: RM = RM[-6000:]; RD = RD[-6000:]
    const = np.array(RM) @ mu
    A = np.array(RD); nc = A.shape[0]
    Db = best[1]
    cvec = np.zeros(nr + 1); cvec[-1] = -1.0
    A_ub = np.hstack([-A, np.ones((nc, 1))]); b_ub = const
    A_eq = np.zeros((1, nr + 1)); A_eq[0, :nr] = 1.0
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[3 * mu.sum()],
                  bounds=[(Db[i] / rho, Db[i] * rho) for i in range(nr)] + [(None, None)], method="highs")
    if res.status != 0:
        print("LP failed", res.message, flush=True); break
    Dt = res.x[:nr]; tmodel = -res.fun
    ft, eigs = evaluate(Sx, mu, Dt, kk)
    rm, rd = cuts_from(Sx, eigs); RM += rm; RD += rd
    acc = ft > best[0] + 1e-6
    if acc:
        best = (ft, Dt.copy()); rho = min(1.0 + (rho - 1.0) * 1.3, 2.0)
        pickle.dump({"D": best[1], "lam": best[0]}, open(out, "wb"))
    else:
        rho = 1.0 + (rho - 1.0) * 0.5
    print("it %2d  trial %.5f  best %.5f  model %.5f  cuts %d  rho %.4f  %s [%.0fs]"
          % (it, ft, best[0], tmodel - 2 / 3, nc, rho, "accepted" if acc else "", time.time() - t0), flush=True)
    if rho < 1.0003: break
print("RESULT cosets %s: unit weights, optimised diagonal: best lambda_min = %.5f (threshold -0.66667); model upper bound %.5f"
      % (sys.argv[1], best[0], tmodel - 2 / 3), flush=True)
