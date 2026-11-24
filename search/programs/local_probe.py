"""Probe for locally adapted (non K'-symmetric) weights around extra vertices.

At a strictly feasible method-B point y (K'-constant weights, target G'' = base - X) compute the dual-type matrix
Z'' = (q/t) (S'')^(-1) on a local edge set L (default: the link edges of the vertices of X, i.e. the edges ab with a, b
neighbours of some x in X), and solve the small problem
     maximise sum_e lam_e   subject to   N^(-1) - diag(lam) >= 0,      N = ((S'')^(-1))_LL,
i.e. how much the diagonal of S'' can be lowered on L alone (both signs allowed) while S'' stays positive
semidefinite.  The gain in beta is sum(lam) / 2 (units a.y = 1).
usage: python3 exp/local_probe.py cosets rm0 y.npy t  v1,v2,..  [mode]
"""
import os, sys, time
import numpy as np
argv = list(sys.argv)
cos_, rm0_, yfile, tval, xs = argv[1], argv[2], argv[3], float(argv[4]), argv[5]
mode = argv[6] if len(argv) > 6 else "link"
sys.argv = ["ipm_b.py", cos_, rm0_, "uniform", "/dev/null", xs]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipm_b.py")).read().split("\nif init == \"uniform\":")[0]
exec(src)
y = np.load(yfile); y = y / (a_vec @ y)
print("beta = %.8f" % (1 + c_vec @ y / 2))
t0 = time.time(); st = State(y); assert st.ok
print("state ok: #neg(S) = %d = #neg(G_FF)  [%.1fs]" % (st.negs, time.time() - t0)); sys.stdout.flush()
K = (st.Q / st.w[None, :]) @ st.Q.T
g = np.fft.irfft(np.conj(st.G), n=q, axis=0)                      # g[d, a, b]:  G[(a,k),(b,l)] = g[(l-k) % q, a, b]
Vr = np.fft.irfft(np.ascontiguousarray(st.Vh.transpose(1, 0, 2)), n=q, axis=1)     # V[(b,l), f] = Vr[b, l, f]
adj = {}
for e in range(NE):
    u, v = int(ends[e, 0]), int(ends[e, 1])
    adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
nb = set()
for x in X: nb |= adj[x]
nb -= set(X)
inL = np.array([ein[e] and (int(ends[e, 0]) in nb) and (int(ends[e, 1]) in nb) for e in range(NE)])
if mode == "ball":      # all target edges with at least one end in N(X)
    inL = np.array([ein[e] and ((int(ends[e, 0]) in nb) or (int(ends[e, 1]) in nb)) for e in range(NE)])
Lidx = np.nonzero(inL)[0]; aL = Lidx // q; kL = Lidx % q; nL = len(Lidx)
print("local edges: %d (mode %s), |N(X)| = %d, |F| = %d" % (nL, mode, len(nb), nF)); sys.stdout.flush()
# diagonal of M'' on all target edges: violation profile
gdiag = g[0][np.arange(nrep), np.arange(nrep)]                    # G[(a,k),(a,k)] = g[0, a, a]
VK = Vr.reshape(NE, nF) @ K
mdiag = np.repeat(gdiag, q) - np.einsum("ef,ef->e", VK, Vr.reshape(NE, nF))
zd = (q / tval) * mdiag
zt = zd[ein]
print("Z''_ee over target edges: mean %.5f  min %.4f  max %.4f  std %.4f" % (zt.mean(), zt.min(), zt.max(), zt.std()))
zl = zd[Lidx]; far = ein & ~inL
print("   on L: mean %.4f min %.4f max %.4f std %.4f;   off L: mean %.5f min %.4f max %.4f std %.4f" % (zl.mean(), zl.min(), zl.max(), zl.std(), zd[far].mean(), zd[far].min(), zd[far].max(), zd[far].std()))
for thr in (0.5, 0.2, 0.1, 0.05, 0.02):
    print("   edges with |Z_ee - 1| > %.2f: %d on L, %d off L" % (thr, int((np.abs(zl - 1) > thr).sum()), int((np.abs(zd[far] - 1) > thr).sum())))
print("   first-order rate sum |Z_ee - 1|: on L %.3f, off L %.3f" % (np.abs(zl - 1).sum(), np.abs(zd[far] - 1).sum()))
sys.stdout.flush()
# N = M''_LL
t0 = time.time()
GLL = g[(kL[None, :] - kL[:, None]) % q, aL[:, None], aL[None, :]]
VL = Vr[aL, kL, :]
N = GLL - VL @ K @ VL.T; N = (N + N.T) / 2
ev = np.linalg.eigvalsh(N)
print("N = M''_LL: eigenvalues min %.3e max %.3e  [%.1fs]" % (ev[0], ev[-1], time.time() - t0)); sys.stdout.flush()
Ninv = np.linalg.inv(N); Ninv = (Ninv + Ninv.T) / 2
evi = np.linalg.eigvalsh(Ninv)
print("N^-1 (Schur complement of S'' on L): eigenvalues min %.3e, median %.3e, max %.3e; diag mean %.4e" % (evi[0], np.median(evi), evi[-1], np.diag(Ninv).mean()))
np.save("scan/probe_Ninv.npy", Ninv)
# small SDP: max sum(lam) s.t. Ninv - diag(lam) >= 0   (log-det barrier, Newton, path following)
lam = np.zeros(nL); tt = 1.0 / np.abs(np.diag(Ninv)).mean() * nL * 10
def val(lam):
    try: Lc = np.linalg.cholesky(Ninv - np.diag(lam))
    except np.linalg.LinAlgError: return None
    return 2 * np.log(np.diag(Lc)).sum()
for stage in range(40):
    for it in range(60):
        A = Ninv - np.diag(lam); W = np.linalg.inv(A); W = (W + W.T) / 2
        gr = -tt * np.ones(nL) + np.diag(W)                      # gradient of  -tt sum(lam) - log det(Ninv - diag lam)
        H = W * W
        dl = -np.linalg.solve(H, gr); dec = -gr @ dl
        if dec < 1e-3: break
        f0 = -tt * lam.sum() - val(lam); al = 1.0
        while True:
            v1 = val(lam + al * dl)
            if v1 is not None and -tt * (lam + al * dl).sum() - v1 <= f0 - 0.2 * al * dec: break
            al *= 0.5
            if al < 1e-9: break
        lam = lam + al * dl
    print("stage %2d  t %.3e  sum(lam) %.6e  gain in beta %.3e  (lam min %.3e max %.3e; %d Newton steps, gap bound %.2e)" % (stage, tt, lam.sum(), lam.sum() / 2, lam.min(), lam.max(), it + 1, nL / tt / 2)); sys.stdout.flush()
    if nL / tt / 2 < 2e-6: break
    tt *= 3
np.save("scan/probe_lam.npy", lam)
