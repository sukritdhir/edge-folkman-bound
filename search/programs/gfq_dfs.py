"""K4-free Cayley graphs on the additive group of GF(q), q = p^k (p odd), connection set S = a union of cosets of a
multiplicative subgroup H0 = <g^r> (closed under negation): complete depth-first search over the K4-free unions and the
unweighted normalised ratio  lambda_min(Deg^(-1/2) M Deg^(-1/2)) / (-2/3)  of every maximal one that is dense enough
(below 1 = the graph arrows (3,3)).  The matrix M is block-diagonalised by the additive characters psi_f(x) =
w^Tr(f x); one frequency per H0-orbit suffices.
usage: python3 exp/gfq_dfs.py q [max_orbits] [budget]         output appended to scan/gfq_dfs.log
"""
import sys, time, os, itertools
import numpy as np

MIN_DENS = float(os.environ.get("MIN_DENS", "0.12"))
REPORT = float(os.environ.get("REPORT", "1.15"))
FROB = bool(int(os.environ.get("FROB", "0"))); MIN_ORB = int(os.environ.get("MIN_ORB", "2"))


def factor_pk(q):
    for p in range(2, q + 1):
        if q % p == 0:
            k = 0; x = q
            while x % p == 0: x //= p; k += 1
            assert x == 1, "q must be a prime power"
            return p, k


class Field:
    def __init__(self, q):
        p, k = factor_pk(q); self.p, self.k, self.q = p, k, q
        digs = np.array([[(a // p ** i) % p for i in range(k)] for a in range(q)], dtype=np.int64)
        pw = p ** np.arange(k)
        self.add = ((digs[:, None, :] + digs[None, :, :]) % p) @ pw            # add[a, b]
        self.neg = ((-digs) % p) @ pw
        self.sub = self.add[:, self.neg]                                      # sub[a, b] = a - b
        # find a monic irreducible polynomial x^k = c_0 + c_1 x + ... for which x is primitive
        found = False
        for coef in itertools.product(range(p), repeat=k):
            if k == 1: break
            if coef[0] == 0: continue
            def mulx(a):
                d = [(a // p ** i) % p for i in range(k)]
                top = d[-1]; nd = [0] + d[:-1]
                nd = [(nd[i] + top * coef[i]) % p for i in range(k)]
                return sum(nd[i] * p ** i for i in range(k))
            e = [1]; a = 1
            for _ in range(q - 2):
                a = mulx(a)
                if a == 1 or a == 0: break
                e.append(a)
            if len(e) == q - 1 and mulx(a) == 1: found = True; break
        if k == 1:
            for gcand in range(2, p):
                e = [1]; a = 1
                for _ in range(q - 2):
                    a = a * gcand % p
                    if a == 1: break
                    e.append(a)
                if len(e) == q - 1: found = True; break
        assert found
        self.exp = np.array(e, dtype=np.int64); self.log = np.full(q, -1, dtype=np.int64); self.log[self.exp] = np.arange(q - 1)
        # trace to the prime field
        tr = np.zeros(q, dtype=np.int64)
        for a in range(1, q):
            s = 0; l = self.log[a]
            for j in range(k): s = self.add[s, self.exp[(l * p ** j) % (q - 1)]]
            assert s < p; tr[a] = s
        self.tr = tr

    def mul(self, a, b):
        """elementwise product of arrays of field elements"""
        a = np.asarray(a); b = np.asarray(b)
        out = self.exp[(self.log[a] + self.log[b]) % (self.q - 1)]
        return np.where((a == 0) | (b == 0), 0, out)


class Cay:
    def __init__(self, F, r, frob=False):
        self.F = F; q = F.q; self.q = q; self.r = r
        half = (q - 1) // 2
        # orbits of the class indices i mod r under negation (i -> i + (q-1)/2) and, with frob, the Frobenius map (i -> p i)
        key = {}
        for i in range(r):
            if i in key: continue
            orb = {i}; stack = [i]
            while stack:
                j = stack.pop()
                nxt = [(j + half) % r] + ([(j * F.p) % r] if frob else [])
                for t in nxt:
                    if t not in orb: orb.add(t); stack.append(t)
            for j in orb: key[j] = min(orb)
        cls = {}
        for i in range(r):
            cls.setdefault(key[i], []).extend(int(x) for x in F.exp[np.arange(i, q - 1, r)])
        self.orbs = [sorted(set(v)) for _, v in sorted(cls.items())]
        self.reps = [int(F.exp[k]) for k in sorted(cls)]
        self.omask = []
        for o in self.orbs:
            mk = np.zeros(q, dtype=bool); mk[o] = True; self.omask.append(mk)
        # one frequency per orbit of the multiplier group (the dual action has the same orbits on the class indices)
        self.freqs = np.array([0] + (self.reps if frob else [int(F.exp[i]) for i in range(r)]))

    def k4_at(self, inS, c):
        F = self.F
        W = inS & inS[F.sub[:, c]]                                  # v in S with v - c in S
        for v in np.nonzero(W)[0]:
            if (W & inS[F.sub[:, v]]).any(): return True
        return False

    def ratio(self, inS):
        F = self.F; q = self.q; p = F.p
        S = np.nonzero(inS)[0]
        if len(S) == 0: return 9.0, 0, 0, 0
        chosen_rep = S < F.neg[S]                                   # one of {c, -c}
        Cp = S[chosen_rep]; m = len(Cp)
        idx = np.full(q, -1, dtype=np.int64); idx[Cp] = np.arange(m); idx[F.neg[Cp]] = np.arange(m)
        isrep = np.zeros(q, dtype=bool); isrep[Cp] = True
        T = inS[F.sub[S[None, :], Cp[:, None]]]                     # (m, d): v - c in S
        deg = T.sum(axis=1)
        if (deg == 0).any(): return 9.0, len(S), 0, 0
        jj, vv = np.nonzero(T); v = S[vv]; c = Cp[jj]
        j2a = idx[v]; x2a = np.where(isrep[v], 0, v)
        w = F.sub[v, c]
        j2b = idx[w]; x2b = np.where(isrep[w], c, v)
        J = np.concatenate([jj, jj]); J2 = np.concatenate([j2a, j2b]); X2 = np.concatenate([x2a, x2b])
        flat = J * m + J2; dm = 1.0 / np.sqrt(deg)
        om = np.exp(2j * np.pi * np.arange(p) / p)
        lam = np.inf
        for f in self.freqs:
            ph = om[F.tr[F.mul(np.full(len(X2), f), X2)]]
            B = (np.bincount(flat, weights=ph.real, minlength=m * m) + 1j * np.bincount(flat, weights=ph.imag, minlength=m * m)).reshape(m, m)
            B = B * dm[:, None] * dm[None, :]
            assert np.abs(B - B.conj().T).max() < 1e-9
            lam = min(lam, np.linalg.eigvalsh(B)[0])
        return float(lam / (-2 / 3)), len(S), int(deg.min()), int(deg.max())


def search(C, budget, out, tag):
    q = C.q; N = len(C.orbs); sizes = [len(o) for o in C.orbs]
    nodes = [0]; best = [None]; nmax = [0]

    def dfs(start, chosen, inS, size):
        nodes[0] += 1
        if nodes[0] > budget: return
        cs = set(chosen)
        ext = [i for i in range(N) if i not in cs and not C.k4_at(inS | C.omask[i], C.reps[i])]
        if ext and size + sum(sizes[i] for i in ext if i >= start) < MIN_DENS * q: return
        if not ext:
            if size >= MIN_DENS * q and size >= 12:
                nmax[0] += 1
                r = C.ratio(inS)
                if best[0] is None or r[0] < best[0][0][0]: best[0] = (r, list(chosen))
                if r[0] < REPORT:
                    line = "q %4d %s  S = union of classes %s  d %3d  tri/edge %d..%d  ratio %.4f %s" % (q, tag, list(chosen), r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
                    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
            return
        for i in ext:
            if i < start: continue
            dfs(i + 1, chosen + [i], inS | C.omask[i], size + sizes[i])
    dfs(0, [], np.zeros(q, dtype=bool), 0)
    return nodes[0], nmax[0], best[0]


if __name__ == "__main__":
    q = int(sys.argv[1]); max_orb = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    F = Field(q); out = open("scan/gfq_dfs.log", "a")
    print("GF(%d): p = %d, k = %d" % (q, F.p, F.k)); sys.stdout.flush()
    sys.setrecursionlimit(10000)
    for r in range(2, q - 1):
        if (q - 1) % r: continue
        C = Cay(F, r, FROB); N = len(C.orbs)
        if not (MIN_ORB <= N <= max_orb): continue
        t0 = time.time()
        nodes, nmax, b = search(C, budget, out, "index %3d (|H0| %3d)%s orbits %2d" % (r, (q - 1) // r, " +Frob" if FROB else "", N))
        line = "# done q %d index %d%s orbits %d  nodes %d%s  maximal %d  best %s  [%.0fs]" % (q, r, "F" if FROB else "", N, nodes, " TRUNCATED" if nodes > budget else "", nmax,
                                                                                     ("ratio %.4f d %d tri/edge %d..%d classes %s" % (b[0][0], b[0][1], b[0][2], b[0][3], b[1])) if b else "none", time.time() - t0)
        print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
