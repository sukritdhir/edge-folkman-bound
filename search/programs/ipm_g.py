"""Interior-point (log-det barrier, Newton) solver for the weighted spectral criterion on a graph with a cyclic
symmetry group of odd order q acting freely on the edges and triangles (generic version of exp/ipm_b.py; the problem
comes from exp/gsym.py).  Variables: symmetric triangle weights mu (per triangle orbit of the base G') and
delta = (2/3) D - Deg(mu) (per edge orbit).  Method A (X = none): S = M_mu + (2/3) D positive definite on the base,
minimise the budget ratio beta = sum D / (3 sum mu) (certificate iff beta < 1).  Method B (extra vertices X): only
the principal submatrix on the edges avoiding X must be positive definite and the budget is counted on G' - X.
See exp/ipm_b.py for the formulas.

usage: python3 exp/ipm_g.py SPEC init.pkl|uniform out.pkl v1,v2,...|none [eps] [gap]
   SPEC: see gsym.from_spec, for example  Y:277:6:3:0:2  or  U:97:6:0
environment: as for exp/ipm_b.py (OMP_NUM_THREADS, POOL, YSTART, T0, KAPPA, TOLC, TOLMID, TSTOP, MAXIT, DSHIFT, ...).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, sys, time, pickle, ctypes
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import zhetrf, zhetri
from scipy.sparse.linalg import LinearOperator, eigsh
sys.path.insert(0, "."); sys.path.insert(0, "exp")
import gsym
from gsym import survival

SPEC = sys.argv[1]
init = sys.argv[2]; outf = sys.argv[3]
A_, perm_, removed_, meta_ = gsym.from_spec(SPEC)
G_, Sx = gsym.make(A_, perm_, removed_, drop_short=meta_.get('drop_short', False))
n = G_.n
X = [] if sys.argv[4] == "none" else [int(v) for v in sys.argv[4].split(",")]
AEXT = bool(os.environ.get("AEXT")) or not X        # method A: the matrix must be positive definite on the whole base
eps = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
gap_target = float(sys.argv[6]) if len(sys.argv) > 6 else 3e-6
q, nrep, no = Sx.q, Sx.nrep, Sx.no; NH = q // 2 + 1; m = no + nrep
oa = np.ascontiguousarray(Sx.oa, dtype=np.int32); oi = np.ascontiguousarray(Sx.oi % q, dtype=np.int32)
cr = np.array([1.0] + [2.0] * (NH - 1))
if q % 2 == 0: cr[-1] = 1.0
assert q % 2 == 1, 'the solver has been validated for odd q only'
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhess.so"))
lib.hess_sym.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 7 + [ctypes.c_int]
lib.hess_sym.restype = None
POOL = int(os.environ.get("POOL", "4")); pool = ThreadPoolExecutor(POOL)

# layout of the base edges: index = orbit * q + offset
ek = np.nonzero(Sx.edge_rep >= 0)[0]
pos = Sx.edge_rep[ek] * q + Sx.edge_off[ek]
NE = nrep * q
ends = np.zeros((NE, 2), dtype=np.int64); ends[pos] = G_.ends[ek]
bad = np.zeros(n, dtype=bool); bad[X] = True
assert all(Sx.keep[x] for x in X), "X must consist of vertices of the base graph"
ein_target = ~bad[ends[:, 0]] & ~bad[ends[:, 1]]
nf, mf = survival(G_, Sx, X)
assert abs(nf.sum() * q - ein_target.sum()) < 1e-6
ein = np.ones(NE, dtype=bool) if AEXT else ein_target        # the edges on which the matrix must be positive definite
F = np.nonzero(~ein)[0]; aF = F // q; kF = F % q; nF = len(F)
c_vec = np.concatenate([nf[oa].sum(axis=1) - 2 * mf, nf]); a_vec = np.concatenate([mf, np.zeros(nrep)])
nubar = float(ein.sum())
print("method %s: base %d vertices, X = %s, |F| = %d, target edges %d, %d variables" % ("A" if AEXT else "B", Sx.nv, X, nF, int(ein_target.sum()), m)); sys.stdout.flush()

dg = np.diag_indices(nrep)
phase = np.exp(-2j * np.pi * np.outer(np.arange(NH), kF) / q)             # phi_r(f) = w^(-r k_f)
iu = np.triu_indices(nF)
pw = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))                          # packing weights (Frobenius inner product)
# assembly of the blocks S_r[a, b] = sum_o mu_o sum_(p, s) [oa_p = a, oa_s = b] w^(r (oi_s - oi_p))  (+ delta on the diagonal)
_pp = np.repeat(np.arange(3), 3); _ss = np.tile(np.arange(3), 3)
pidx = (oa[:, _pp].astype(np.int64) * nrep + oa[:, _ss]).ravel()
doff = ((oi[:, _ss].astype(np.int64) - oi[:, _pp]) % q)
wtab = np.exp(2j * np.pi * np.outer(np.arange(NH), np.arange(q)) / q)


def block(r, mu, dl):
    ph = wtab[r][doff] * mu[:, None]
    out = np.bincount(pidx, weights=ph.real.ravel(), minlength=nrep * nrep) + 1j * np.bincount(pidx, weights=ph.imag.ravel(), minlength=nrep * nrep)
    out = out.reshape(nrep, nrep); out[dg] += dl
    return out


def blocks(mu, dl):
    return np.array(list(pool.map(lambda r: block(r, mu, dl), range(NH))))


def inertia(ldu, ipiv):
    """number of negative eigenvalues and log|det| from a Bunch-Kaufman factorisation (lower)"""
    d = np.real(np.diag(ldu)); sub = np.diag(ldu, -1)
    neg = 0; ld = 0.0; k = 0; N = len(ipiv)
    while k < N:
        if ipiv[k] > 0:
            neg += d[k] < 0; ld += np.log(abs(d[k])); k += 1
        else:
            det = d[k] * d[k + 1] - abs(sub[k]) ** 2
            if det < 0: neg += 1
            elif d[k] + d[k + 1] < 0: neg += 2
            ld += np.log(abs(det)); k += 2
    return int(neg), ld


class State:
    """everything derived from y that the barrier needs"""
    def __init__(self, y):
        self.y = y; self.ok = False
        mu = y[:no]; dl = y[no:]

        def work(r):
            Sr = block(r, mu, dl)
            ldu, ipiv, info = zhetrf(Sr, lower=1, overwrite_a=1)
            if info != 0: return None
            ng, ld = inertia(ldu, ipiv)
            gi, info = zhetri(ldu, ipiv, lower=1, overwrite_a=1)
            if info != 0: return None
            gi = np.tril(gi); Gr = gi + np.tril(gi, -1).conj().T
            Vr = Gr[:, aF] * phase[r][None, :]
            gff = (cr[r] / q) * np.real(np.conj(phase[r])[:, None] * Vr[aF, :])
            return ng, ld, Gr, Vr, gff
        res = list(pool.map(work, range(NH)))
        if any(x is None for x in res): return
        self.negs = int(sum(int(cr[r]) * res[r][0] for r in range(NH)))
        lds = float(sum(cr[r] * res[r][1] for r in range(NH)))
        self.G = np.array([x[2] for x in res]); self.Vh = np.array([x[3] for x in res])
        if nF == 0:
            del res
            self.negF = 0; self.ok = (self.negs == 0); self.logdet = lds; self.w = np.zeros(0); self.Q = np.zeros((0, 0)); return
        GFF = sum(x[4] for x in res); del res
        GFF = (GFF + GFF.T) / 2
        w, Q = np.linalg.eigh(GFF)
        self.negF = int((w < 0).sum())
        if np.abs(w).min() < 1e-13 * np.abs(w).max(): return
        self.ok = (self.negs == self.negF)
        self.logdet = lds + np.log(np.abs(w)).sum()                       # log det S''
        self.w = w; self.Q = Q

    def dirderiv(self, dS):
        """tr((S'')^(-1) dS'') for a K'-symmetric direction given by its blocks dS (cheap; used by the line search)"""
        K = (self.Q / self.w[None, :]) @ self.Q.T

        def work(r):
            t1 = np.real(np.vdot(dS[r], self.G[r]))                       # sum conj(dS) * G = tr(dS^H G) = tr(G dS)
            Mr = self.Vh[r].conj().T @ (dS[r] @ self.Vh[r])
            return cr[r] * (t1 - np.real(np.sum(K * Mr.T)) / q)
        return float(sum(pool.map(work, range(NH))))

    def derivatives(self):
        """gradient and Hessian of Phi = -log det S''"""
        t0 = time.time()
        G = self.G; Vh = self.Vh; w = self.w; Q = self.Q
        H = np.empty((m, m)); g = np.empty(m)
        lib.hess_sym(nrep, no, NH, q, G.ctypes.data, None, oa.ctypes.data, oi.ctypes.data, cr.ctypes.data, H.ctypes.data, g.ctypes.data, 0)
        t1 = time.time()
        if nF == 0:
            self.times = (t1 - t0, 0.0, 0.0, 0.0)
            return H, g
        K = (Q / w[None, :]) @ Q.T
        Rb = np.array(list(pool.map(lambda r: (Vh[r] @ K) @ Vh[r].conj().T / q, range(NH))))
        H2 = np.empty((m, m))
        lib.hess_sym(nrep, no, NH, q, G.ctypes.data, Rb.ctypes.data, oa.ctypes.data, oi.ctypes.data, cr.ctypes.data, H2.ctypes.data, None, 0)
        H -= 2 * H2; del H2, Rb
        t2 = time.time()
        # V in real space: V[(b, l), f] = (1/q) sum_r Vh_r[b, f] w^(r l)
        Vr = np.fft.irfft(np.ascontiguousarray(Vh.transpose(1, 0, 2)), n=q, axis=1)          # (nrep, q, nF)
        L = Q / np.sqrt(np.abs(w))[None, :]                                # K = L diag(sign w) L^T
        VL = Vr @ L; del Vr                                                 # (nrep, q, nF)
        sg = np.sign(w)
        sh = np.arange(q)
        CH = 64
        # third term  sum_(f <= f') s_f s_f' pw^2 C_i[f,f'] C_j[f,f'],  C_i = sum over the orbit of b b^T  (b = rows of VL summed
        # over the edges of a triangle).  The pairs (f, f') are processed in blocks of FB rows f, so that the memory stays small.
        FB = int(os.environ.get("FBLOCK", "48"))
        tF = 0.0; tG = 0.0
        rows = [(i * m // POOL, (i + 1) * m // POOL) for i in range(POOL)]
        first = [True]
        for f0 in range(0, nF, FB):
            f1 = min(nF, f0 + FB); tb = time.time()
            ff, fp = np.nonzero(np.arange(f0, f1)[:, None] <= np.arange(nF)[None, :])     # pairs (f0 + ff, fp) with f <= f'
            wcol = np.where(ff + f0 == fp, 1.0, np.sqrt(2.0))
            scol = sg[ff + f0] * sg[fp]
            P = np.empty((m, len(ff)), dtype=np.float64)

            def tri_chunk(i0):
                i1 = min(no, i0 + CH)
                B = np.zeros((i1 - i0, q, nF))
                for p in range(3):
                    B += VL[oa[i0:i1, p][:, None], (oi[i0:i1, p][:, None] + sh[None, :]) % q, :]
                C = np.matmul(B[:, :, f0:f1].transpose(0, 2, 1), B)
                P[i0:i1] = C[:, ff, fp] * wcol[None, :]
                if first[0]: g[i0:i1] += np.einsum("ikf,ikf,f->i", B, B, sg)

            def edge_chunk(a0):
                a1 = min(nrep, a0 + CH)
                B = VL[a0:a1]
                C = np.matmul(B[:, :, f0:f1].transpose(0, 2, 1), B)
                P[no + a0:no + a1] = C[:, ff, fp] * wcol[None, :]
                if first[0]: g[no + a0:no + a1] += np.einsum("ikf,ikf,f->i", B, B, sg)
            list(pool.map(tri_chunk, range(0, no, CH))); list(pool.map(edge_chunk, range(0, nrep, CH)))
            first[0] = False
            tF += time.time() - tb; tb = time.time()
            neg_c = np.nonzero(scol < 0)[0]
            Pn = np.ascontiguousarray(P[:, neg_c]) if len(neg_c) else None
            if Pn is not None: P[:, neg_c] = 0.0

            def gram(rg):
                i0, i1 = rg
                H[i0:i1] += P[i0:i1] @ P.T
                if Pn is not None: H[i0:i1] -= Pn[i0:i1] @ Pn.T
            list(pool.map(gram, rows))
            del P, Pn
            tG += time.time() - tb
        del VL
        t3 = t2 + tF
        self.times = (t1 - t0, t2 - t1, tF, tG)
        return H, g


def apply_blocks(Mb, x):
    """product of the K'-symmetric matrix with blocks Mb and a real vector in the edge layout"""
    Xh = np.fft.rfft(x.reshape(nrep, q), axis=1)
    Yh = np.empty_like(Xh)

    def work(r): Yh[:, r] = Mb[r] @ Xh[:, r]
    list(pool.map(work, range(NH)))
    return np.fft.irfft(Yh, n=q, axis=1).ravel()


Eidx = np.nonzero(ein)[0]


def boundary_step(st, Sb, dSb):
    """largest alpha with S''(y + alpha Dl) >= 0: -1 / (smallest eigenvalue of the pencil (dS'', S'')), by Lanczos.
    S'' x = (S x)_E for x supported on E;  (S'')^(-1) b = [G (b - E_F K (G b)_F)]_E."""
    K = (st.Q / st.w[None, :]) @ st.Q.T
    buf = np.zeros(NE)

    def emb(x):
        v = np.zeros(NE); v[Eidx] = x; return v

    def opA(x): return apply_blocks(dSb, emb(x))[Eidx]

    def opM(x): return apply_blocks(Sb, emb(x))[Eidx]

    def opMinv(b):
        v = emb(b); u = apply_blocks(st.G, v); v[F] = -(K @ u[F])
        return apply_blocks(st.G, v)[Eidx]
    nE = len(Eidx)
    A = LinearOperator((nE, nE), matvec=opA, dtype=float); M = LinearOperator((nE, nE), matvec=opM, dtype=float)
    Mi = LinearOperator((nE, nE), matvec=opMinv, dtype=float)
    try:
        e = eigsh(A, k=1, M=M, Minv=Mi, which="SA", tol=1e-4, ncv=24, maxiter=400, return_eigenvectors=False)
        emin = float(e[0])
    except Exception as ex:
        print("    (Lanczos for the boundary step failed: %s)" % str(ex)[:80]); return None
    return (-1.0 / emin) if emin < 0 else np.inf


def dense_full(y):
    """the full matrix S in the edge layout (small graphs only)"""
    S = blocks(y[:no], y[no:])
    full = np.zeros((NE, NE)); idx = np.arange(q)
    for a in range(nrep):
        for b in range(nrep):
            sab = np.zeros(q)
            for r in range(NH): sab += cr[r] * np.real(S[r][a, b] * np.exp(-2j * np.pi * r * idx / q)) / q
            full[a * q + idx[:, None], b * q + idx[None, :]] = sab[(idx[None, :] - idx[:, None]) % q]      # S[(a,k),(b,l)] = s_ab(l - k)
    return full


def dense_logdet(y):
    """log det S'' by dense linear algebra (small graphs only)"""
    full = dense_full(y)
    ev = np.linalg.eigvalsh(full[np.ix_(ein, ein)])
    return float(np.log(np.abs(ev)).sum()), int((ev < 0).sum()), int((np.linalg.eigvalsh(full) < 0).sum())


def save(y, tag):
    mu = y[:no]; dl = y[no:]
    D = 1.5 * (dl + Sx.degrees(mu))
    beta = (nf * D).sum() / (3 * (mf * mu).sum())
    Dt = D / beta
    pickle.dump({"mu": mu, "D": Dt, "beta": beta, "X": list(X), "extra": list(X), "spec": SPEC, "meta": meta_, "method": "A" if AEXT else "B"}, open(outf, "wb"))
    print("   saved %s: beta = %.8f, min mu/mean = %.4f, min D/mean = %.4f %s" % (outf, beta, mu.min() / mu.mean(), Dt.min() / Dt.mean(), tag)); sys.stdout.flush()


if init == "uniform":
    mu = np.ones(no); D = Sx.degrees(mu)
    if os.environ.get("DSHIFT"): D = D + float(os.environ["DSHIFT"]) * D.mean()      # tests on small graphs with triangle-free edges
else:
    d0 = pickle.load(open(init, "rb")); mu = np.array(d0["mu"], float)
    D = np.array(d0["D"], float) if d0.get("D") is not None else Sx.degrees(mu)
s = (mf * mu).sum(); mu = mu / s; D = D / s
D *= 3.0 / (nf * D).sum()
D *= (1 + eps)
y = np.concatenate([mu, (2.0 / 3.0) * D - Sx.degrees(mu)])
if os.environ.get("YSTART"):
    y = np.load(os.environ["YSTART"]); y = y / (a_vec @ y)
print("start: nubar = %.0f, beta = %.6f" % (nubar, 1 + c_vec @ y / 2)); sys.stdout.flush()
t_ = time.time(); st = State(y)
assert st.ok, "the starting point is not strictly feasible (neg S %s, neg G_FF %s): increase eps" % (getattr(st, "negs", "?"), getattr(st, "negF", "?"))
print("start: #neg(S) = %d = #neg(G_FF), log det = %.4f  [state %.1fs]" % (st.negs, st.logdet, time.time() - t_)); sys.stdout.flush()

if os.environ.get("DENSETEST"):
    ld, ng, ngfull = dense_logdet(y)
    print("dense: log det S'' = %.6f (neg %d; full S has %d negative eigenvalues); formula: %.6f" % (ld, ng, ngfull, st.logdet))
if os.environ.get("GRADTEST"):
    # coordinate-wise check of the gradient by central differences, using the dense log det when DENSETEST is set
    H, g = st.derivatives()
    print("Euler identity: y.g = %.6f (should be -%d);  y.H.y = %.6f (should be +%d); |H y + g| / |g| = %.3e" % (y @ g, int(ein.sum()), y @ H @ y, int(ein.sum()), np.linalg.norm(H @ y + g) / np.linalg.norm(g)))
    rng = np.random.default_rng(5)
    idx = list(rng.choice(no, 4, replace=False)) + [no + int(a) for a in np.unique(aF)[:3]] + [no + int(a) for a in rng.choice(nrep, 3, replace=False)]
    for i in idx:
        hstep = float(os.environ["GRADTEST"]) * max(abs(y[i]), 1e-12)
        e = np.zeros(m); e[i] = hstep
        sp_, sm_ = State(y + e), State(y - e)
        line = "var %4d (%s%s): g = %+.6e, formula FD %+.6e" % (i, "mu" if i < no else "delta", " F-orbit" if i >= no and (i - no) in set(aF.tolist()) else "", g[i], -(sp_.logdet - sm_.logdet) / (2 * hstep))
        if os.environ.get("DENSETEST"):
            line += ", dense FD %+.6e" % (-(dense_logdet(y + e)[0] - dense_logdet(y - e)[0]) / (2 * hstep))
        line += "  [inertia %d/%d -> %d/%d, %d/%d]" % (st.negs, st.negF, sp_.negs, sp_.negF, sm_.negs, sm_.negF)
        print(line); sys.stdout.flush()
    sys.exit(0)
if os.environ.get("FDTEST"):
    H, g = st.derivatives()
    print("derivative times: sym %.1fs, cross %.1fs, F-part %.1fs, Gram %.1fs" % st.times)
    rng = np.random.default_rng(1); v = rng.standard_normal(m) * np.abs(y) * float(os.environ.get("FDSCALE", "1e-5"))
    st2 = State(y + v); H2, g2 = st2.derivatives()
    print("FD test: |g2 - g - H v| / |H v| = %.2e;   (Phi2 - Phi) - (g + g2).v/2 = %.3e (Phi change %.3e)"
          % (np.linalg.norm(g2 - g - H @ v) / np.linalg.norm(H @ v), -(st2.logdet - st.logdet) - 0.5 * (g + g2) @ v, -(st2.logdet - st.logdet)))
    dSb = blocks(v[:no], v[no:])
    print("directional derivative: %.8e, g.v = %.8e" % (-st.dirderiv(dSb), g @ v))
    print("Euler identity: y.g = %.6f (should be -%d); |H y + g| / |g| = %.3e; symmetry of H: %.2e"
          % (y @ g, int(ein.sum()), np.linalg.norm(H @ y + g) / np.linalg.norm(g), np.abs(H - H.T).max() / np.abs(H).max()))
    print("inertia: #neg(S) = %d, #neg(G_FF) = %d; G_FF eigenvalues: min |w| %.3e, max |w| %.3e" % (st.negs, st.negF, np.abs(st.w).min(), np.abs(st.w).max()))
    sys.exit(0)

t = None; kappa = float(os.environ.get("KAPPA", "5")); it = 0; ncen = 0; t00 = time.time()
tol_c = float(os.environ.get("TOLC", "0.9")); TOL_MID = float(os.environ.get("TOLMID", "1.0"))
TSTOP = float(os.environ.get("TSTOP", "1e99")); FRAC = float(os.environ.get("FRAC", "0.9"))
THETA = float(os.environ.get("THETA", "0.5")); KMAX = float(os.environ.get("KMAX", "3"))
al_prev = 0.25
while True:
    it += 1; t0 = time.time()
    H, gphi = st.derivatives()
    euler = max(abs(y @ gphi + nubar) / nubar, np.linalg.norm(H @ y + gphi) / np.linalg.norm(gphi))      # accuracy indicator (both are 0 in exact arithmetic)
    t1 = time.time()
    sc = 1.0 / np.sqrt(np.diag(H))
    H *= sc[:, None]; H *= sc[None, :]
    reg = 0.0
    while True:
        try:
            cf = cho_factor(H if reg == 0.0 else H + reg * np.eye(m), lower=True, check_finite=False); break
        except np.linalg.LinAlgError:
            reg = 1e-13 if reg == 0.0 else reg * 10

    def hsolve(b):
        x = cho_solve(cf, sc * b)
        if reg > 0:
            for _ in range(2): x += cho_solve(cf, sc * b - H @ x)
        return sc * x
    v_c = hsolve(c_vec); v_a = hsolve(a_vec); v_g = hsolve(gphi)
    del H
    if t is None:
        A2 = np.array([[c_vec @ v_c, c_vec @ v_a], [c_vec @ v_a, a_vec @ v_a]]); b2 = -np.array([c_vec @ v_g, a_vec @ v_g])
        t = float(np.linalg.solve(A2, b2)[0])
        if os.environ.get("T0"): t = float(os.environ["T0"])
        print("initial t = %.4g (gap bound nubar/2t = %.2e)" % (t, nubar / (2 * t))); sys.stdout.flush()
        if t <= 0: t = nubar / (2 * 0.01)

    def direction(t):
        nu = -(t * (a_vec @ v_c) + a_vec @ v_g) / (a_vec @ v_a)
        Dl = -(t * v_c + v_g + nu * v_a)
        return Dl, float(-(t * c_vec + gphi) @ Dl), nu
    Dl, lam2, nu = direction(t)
    msg = ""
    last = nubar / (2 * t) < gap_target
    if lam2 < (tol_c if last else TOL_MID):
        beta = 1 + c_vec @ y / 2; gap = beta - (1 - nu / (2 * t))
        print("CENTRAL t = %.4g: beta = %.8f, lower bound on min beta about %.8f (rigorous only if lam2 = %.2f < 1); margin now %+.7f  (%d Newton steps, %.0fs total)"
              % (t, beta, beta - gap, lam2, (2 / 3) * (1 - beta), ncen, time.time() - t00)); sys.stdout.flush()
        save(y, "")
        if last or t > TSTOP: break
        if THETA > 0:
            # adaptive step in t: the first Newton step after the increase is the tangent of the central path; choose the
            # increase so that this step stops at the fraction THETA of the distance to the boundary
            nu1 = -(a_vec @ v_c) / (a_vec @ v_a); d1 = -(v_c + nu1 * v_a)
            sbd = boundary_step(st, blocks(y[:no], y[no:]), blocks(d1[:no], d1[no:]))
            if sbd is not None and np.isfinite(sbd): t_new = min(max(t + THETA * sbd, 1.05 * t), KMAX * t)
            else: t_new = t * min(kappa, KMAX)
            msg = "t -> %.4g (factor %.3f, boundary of the tangent at factor %s)" % (t_new, t_new / t, ("%.3f" % (1 + sbd / t)) if sbd is not None and np.isfinite(sbd) else "inf")
            t = t_new; ncen = 0
        else:
            if ncen <= 3: kappa = min(kappa * 1.5, 100.0)
            elif ncen > 7: kappa = max(kappa / 1.5, 1.5)
            t *= kappa; ncen = 0; msg = "t -> %.3g" % t
        Dl, lam2, nu = direction(t)
    t2 = time.time()
    # line search on f(alpha) = t c.(y + alpha Dl) - log det S''(y + alpha Dl); f'(0) = -lam2.  A step is accepted when it is
    # feasible, decreases f enough (Armijo) and is not far beyond the minimiser along the line (f'(alpha) <= 0.1 lam2): the last
    # test keeps the iterates away from the boundary, where the formulas through S^(-1) lose their accuracy.
    f0 = t * (c_vec @ y) - st.logdet
    dSb = blocks(Dl[:no], Dl[no:]); cD = t * (c_vec @ Dl)
    fp0 = cD - st.dirderiv(dSb)
    al_bd = None
    if lam2 >= 1 and not os.environ.get("NOLANCZOS"):
        tb = time.time(); al_bd = boundary_step(st, blocks(y[:no], y[no:]), dSb); tb = time.time() - tb
    if al_bd is not None: al = min(1.0, FRAC * al_bd)
    else: al = 1.0 if lam2 < 1 else min(1.0, 3.0 * al_prev)
    ntry = 0; best = None; hi = None
    while True:
        ntry += 1
        s2 = State(y + al * Dl)
        fp = None
        if s2.ok:
            f1 = t * (c_vec @ (y + al * Dl)) - s2.logdet
            fp = cD - s2.dirderiv(dSb)
            if f1 <= f0 - 0.2 * al * lam2 and fp <= 0.1 * lam2: best = (al, s2, f1, fp); break
        hi = al
        if fp is not None and fp > 0:
            al *= max(0.25, min(0.7, lam2 / (lam2 + fp)))                 # secant estimate of the minimiser, safeguarded
        else:
            al *= 0.5
        if al < 1e-7: break
    if best is None:
        print("line search failed (lam2 %.3e); stopping" % lam2); break
    if ntry == 1 and best[3] < -0.5 * lam2 and best[0] < 1.0 and al_bd is None:   # still steep: try a longer step once
        al2 = min(1.0, 2.5 * best[0]); s2 = State(y + al2 * Dl); ntry += 1
        if s2.ok:
            f2 = t * (c_vec @ (y + al2 * Dl)) - s2.logdet; fp2 = cD - s2.dirderiv(dSb)
            if f2 < best[2] and fp2 <= 0.1 * lam2: best = (al2, s2, f2, fp2)
    tms = st.times
    al, st, f1, fp = best; y = st.y; ncen += 1; al_prev = al
    np.save(outf + ".y.npy", y)
    print("it %3d  t %.3g  beta %.8f  lam2 %.3e (f'(0) %.3e)  alpha %.4f (%d trials, f' %.2e, boundary %s)  f %.4f  #neg(S) %d  min mu/mean %.3f  euler %.1e  [deriv %.0fs (sym %.0f, cross %.0f, F %.0f, Gram %.0f), solve %.0fs, search %.0fs] reg %.0e %s"
          % (it, t, 1 + c_vec @ y / 2, lam2, fp0, al, ntry, fp, ("%.4g" % al_bd) if al_bd is not None else "-", f1, st.negs, y[:no].min() / y[:no].mean(), euler, t1 - t0, tms[0], tms[1], tms[2], tms[3], t2 - t1, time.time() - t2, reg, msg)); sys.stdout.flush()
    if os.environ.get("DEBUGDENSE"):
        full = dense_full(y); ev2 = np.linalg.eigvalsh(full[np.ix_(ein, ein)]); ev = np.linalg.eigvalsh(full)
        print("    dense: S'' smallest", np.array2string(ev2[:4], precision=3), " S smallest", np.array2string(ev[:4], precision=3), " G_FF |w| min %.2e max %.2e" % (np.abs(st.w).min(), np.abs(st.w).max()))
    if it >= int(os.environ.get("MAXIT", "100000")): break
print("finished in %.0fs" % (time.time() - t00))
