"""K4-free Cayley graphs on the additive group of a finite commutative ring R = F_1 x ... x F_r, where each F_i is a
field GF(p^k) or a ring Z_m, with connection set S = a union of orbits of a subgroup M of the unit group R^* (acting by
multiplication, -1 in M).  This contains the circulants with a multiplier group (R = Z_n) and the cyclotomic graphs
over GF(q); the new cases are the mixed products, e.g. GF(9) x Z_73, GF(23) x GF(23), GF(25) x GF(27).
Complete depth-first search over the K4-free unions; unweighted normalised ratio
     lambda_min(Deg^(-1/2) M Deg^(-1/2)) / (-2/3)        (below 1 = the graph arrows (3,3))
of every maximal union of density >= MIN_DENS, computed with the characters of the additive group.
Elements x with 2x = 0 are never put into S (the ratio code pairs c with -c).
usage: python3 exp/ring_dfs.py lo hi out.log [maxorb] [budget]      (all rings with lo <= |R| <= hi that are neither
       Z_n nor a single field), or  python3 exp/ring_dfs.py spec out.log [maxorb] [budget]  with spec like F9xZ73.
env: MIN_DENS (0.14), REPORT (1.08), MAXGROUPS (400 groups per ring, those with the fewest orbits first)
"""
import sys, time, os, itertools
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
import numpy as np
from gfq_dfs import Field

MIN_DENS = float(os.environ.get("MIN_DENS", "0.14"))
REPORT = float(os.environ.get("REPORT", "1.08"))
MAXGROUPS = int(os.environ.get("MAXGROUPS", "400"))


class Ring:
    def __init__(self, spec):
        """spec: list of ('F', q) or ('Z', m)"""
        self.spec = spec; self.name = "x".join("%s%d" % s for s in spec)
        add = np.zeros((1, 1), dtype=np.int64); mul = np.zeros((1, 1), dtype=np.int64); neg = np.zeros(1, dtype=np.int64)
        char = np.ones((1, 1), dtype=complex); one = 0
        for kind, q in spec:
            idx = np.arange(q)
            isprime = q > 1 and all(q % r for r in range(2, int(q ** 0.5) + 1))
            if kind == "F" and not isprime:
                F = Field(q); a2 = F.add; n2 = F.neg; m2 = F.mul(idx[:, None], idx[None, :]); c2 = np.exp(2j * np.pi * F.tr[m2] / F.p); o2 = 1
            else:
                a2 = (idx[:, None] + idx[None, :]) % q; n2 = (-idx) % q; m2 = (idx[:, None] * idx[None, :]) % q
                c2 = np.exp(2j * np.pi * m2 / q); o2 = 1 % q
            n1 = len(neg)
            add = (add[:, None, :, None] * q + a2[None, :, None, :]).reshape(n1 * q, n1 * q)
            mul = (mul[:, None, :, None] * q + m2[None, :, None, :]).reshape(n1 * q, n1 * q)
            neg = (neg[:, None] * q + n2[None, :]).reshape(-1)
            char = (char[:, None, :, None] * c2[None, :, None, :]).reshape(n1 * q, n1 * q)
            one = one * q + o2
        self.n = len(neg); self.add = add; self.mul = mul; self.neg = neg; self.char = char; self.one = one
        self.sub = add[:, neg]                                              # sub[a, b] = a - b
        self.units = [int(x) for x in range(self.n) if (mul[x] == one).any()]
        self.invol = set(int(x) for x in range(1, self.n) if add[x, x] == 0)

    def subgroups(self, cap=20000):
        """subgroups of R^* containing -1, as frozensets"""
        mul = self.mul
        def gen(M, s):
            out = set(M); x = s
            while x not in out:
                out |= {int(mul[x, a]) for a in M}
                x = int(mul[x, s])
            return frozenset(out)
        base = gen(frozenset([self.one]), int(self.neg[self.one]))
        cyc = set(gen(base, s) for s in self.units)
        allg = set(cyc); frontier = list(cyc)
        while frontier and len(allg) < cap:
            new = []
            for M in frontier:
                for C in cyc:
                    if C <= M: continue
                    J = M
                    for c in C:
                        if c not in J: J = gen(J, c)
                    if J not in allg: allg.add(J); new.append(J)
                    if len(allg) >= cap: break
                if len(allg) >= cap: break
            frontier = new
        return allg

    def orbits(self, M, zero=False):
        mul = self.mul; seen = np.zeros(self.n, dtype=bool); orbs = []; Ml = np.array(sorted(M))
        for x in range(0 if zero else 1, self.n):
            if seen[x]: continue
            o = np.unique(mul[Ml, x]); seen[o] = True; orbs.append([int(y) for y in o])
        return orbs


def ratio(R, inS, freqs):
    n = R.n; sub = R.sub; neg = R.neg
    S = np.nonzero(inS)[0]
    if len(S) == 0: return 9.0, 0, 0, 0
    Cp = S[S < neg[S]]; m = len(Cp)
    idx = np.full(n, -1, dtype=np.int64); idx[Cp] = np.arange(m); idx[neg[Cp]] = np.arange(m)
    isrep = np.zeros(n, dtype=bool); isrep[Cp] = True
    T = inS[sub[S[None, :], Cp[:, None]]]                           # (m, d): v - c in S
    deg = T.sum(axis=1)
    if (deg == 0).any(): return 9.0, len(S), 0, int(deg.max())
    jj, vv = np.nonzero(T); v = S[vv]; c = Cp[jj]
    j2a = idx[v]; x2a = np.where(isrep[v], 0, v)
    w = sub[v, c]
    j2b = idx[w]; x2b = np.where(isrep[w], c, v)
    J = np.concatenate([jj, jj]); J2 = np.concatenate([j2a, j2b]); X2 = np.concatenate([x2a, x2b])
    flat = J * m + J2; dm = 1.0 / np.sqrt(deg)
    lam = np.inf
    for f in freqs:
        ph = R.char[f, X2]
        B = (np.bincount(flat, weights=ph.real, minlength=m * m) + 1j * np.bincount(flat, weights=ph.imag, minlength=m * m)).reshape(m, m)
        B = B * dm[:, None] * dm[None, :]
        assert np.abs(B - B.conj().T).max() < 1e-8
        lam = min(lam, np.linalg.eigvalsh(B)[0])
    return float(lam / (-2 / 3)), len(S), int(deg.min()), int(deg.max())


def bits(arr):
    return int.from_bytes(np.packbits(arr, bitorder="little").tobytes(), "little")


def search(R, M, orbs, budget, out, tag):
    n = R.n
    orbs = [o for o in orbs if not (set(o) & R.invol)]
    N = len(orbs); sizes = [len(o) for o in orbs]; reps = [o[0] for o in orbs]
    if N < 1 or sum(sizes) < MIN_DENS * n: return 0, 0, None
    freqs = [o[0] for o in R.orbits(M, zero=True)]
    # B[i][v] = bitset of O_i + v
    B = []
    for o in orbs:
        A = np.zeros((n, n), dtype=bool)
        A[np.repeat(np.arange(n), len(o)), R.add[:, o].reshape(-1)] = True      # row v: the elements s + v, s in O_i
        B.append([bits(A[v]) for v in range(n)])
    obits = [B[i][0] for i in range(N)]
    nodes = [0]; best = [None]; nmax = [0]

    def dfs(start, chosen, Sb, TS, size):
        nodes[0] += 1
        if nodes[0] > budget: return
        cs = set(chosen); ext = []
        for i in range(N):
            if i in cs: continue
            c = reps[i]; Bi = B[i]
            W = (Sb | obits[i]) & (TS[c] | Bi[c]); w = W; bad = False
            while w:
                v = (w & -w).bit_length() - 1
                if W & (TS[v] | Bi[v]): bad = True; break
                w &= w - 1
            if not bad: ext.append(i)
        if ext and size + sum(sizes[i] for i in ext if i >= start) < MIN_DENS * n: return
        if not ext:
            if size >= MIN_DENS * n and size >= 12:
                nmax[0] += 1
                inS = np.zeros(n, dtype=bool)
                for i in chosen: inS[orbs[i]] = True
                r = ratio(R, inS, freqs)
                if best[0] is None or r[0] < best[0][0][0]: best[0] = (r, [reps[i] for i in chosen])
                if r[0] < REPORT:
                    line = "R %s (n %d) %s  S = union of orbits of %s  d %3d  tri/edge %d..%d  ratio %.4f %s" % (R.name, n, tag, [reps[i] for i in chosen], r[1], r[2], r[3], r[0], "ARROWS" if r[0] < 1 else "")
                    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()
            return
        for i in ext:
            if i < start: continue
            Bi = B[i]
            dfs(i + 1, chosen + [i], Sb | obits[i], [a | b for a, b in zip(TS, Bi)], size + sizes[i])
    dfs(0, [], 0, [0] * n, 0)
    return nodes[0], nmax[0], best[0]


def factor(n):
    f = {}; d = 2
    while d * d <= n:
        while n % d == 0: f[d] = f.get(d, 0) + 1; n //= d
        d += 1
    if n > 1: f[n] = f.get(n, 0) + 1
    return f


def partitions(a, mx=None):
    if a == 0: yield []; return
    for k in range(min(a, mx or a), 0, -1):
        for rest in partitions(a - k, k): yield [k] + rest


def rings_of_order(n):
    """all specs of order n, except Z_n itself and a single field"""
    f = factor(n); per_prime = []
    for p, a in sorted(f.items()):
        opts = []
        for part in partitions(a):
            # each part k: field GF(p^k), or Z_{p^k} when k >= 2
            choices = [([("F", p ** k)] if k == 1 else [("F", p ** k), ("Z", p ** k)]) for k in part]
            for ch in itertools.product(*choices):
                key = tuple(sorted(ch))
                if key not in opts: opts.append(key)
        per_prime.append(opts)
    out = []
    for combo in itertools.product(*per_prime):
        spec = [s for part in combo for s in part]
        cyclic = all(len(part) == 1 and (part[0][0] == "Z" or part[0][1] in f) for part in combo)   # Z_n
        single_field = len(spec) == 1 and spec[0][0] == "F"
        if cyclic or single_field: continue
        out.append(spec)
    return out


def run_ring(spec, out, maxorb, budget):
    t0 = time.time(); R = Ring(spec); n = R.n
    groups = R.subgroups()
    cand = []
    for M in groups:
        if (n - 1) / len(M) > maxorb: continue
        orbs = R.orbits(M)
        if 2 <= len(orbs) <= maxorb: cand.append((len(orbs), -len(M), M, orbs))
    cand.sort(key=lambda t: (t[0], t[1]))
    ng = 0; bestn = None; trunc = 0
    for (no, _, M, orbs) in cand[:MAXGROUPS]:
        ng += 1
        nodes, nmax, b = search(R, M, orbs, budget, out, "|M| %3d orbits %2d" % (len(M), no))
        if nodes > budget: trunc += 1
        if b is not None and (bestn is None or b[0][0] < bestn[0]): bestn = (b[0][0], b[0][1], b[0][2], b[0][3], len(M), no, b[1])
    line = "# done %s n %d  subgroups %d  with 2..%d orbits %d  searched %d  truncated %d  best %s  [%.0fs]" % (
        R.name, n, len(groups), maxorb, len(cand), ng, trunc, ("ratio %.4f d %d tri/edge %d..%d |M| %d orbits %d reps %s" % bestn) if bestn else "none", time.time() - t0)
    print(line); out.write(line + "\n"); out.flush(); sys.stdout.flush()


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    if sys.argv[1][0] in "FZ":
        spec = [(s[0], int(s[1:])) for s in sys.argv[1].split("x")]
        out = open(sys.argv[2], "a"); maxorb = int(sys.argv[3]) if len(sys.argv) > 3 else 30; budget = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
        run_ring(spec, out, maxorb, budget)
    else:
        lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = open(sys.argv[3], "a")
        maxorb = int(sys.argv[4]) if len(sys.argv) > 4 else 30; budget = int(sys.argv[5]) if len(sys.argv) > 5 else 50000
        done = set(l.split()[2] for l in open(sys.argv[3]) if l.startswith("# done"))
        par = os.environ.get("PARITY", "")                                   # "odd" or "even" restricts n
        for n in range(hi, lo - 1, -1):
            if (par == "odd" and n % 2 == 0) or (par == "even" and n % 2 == 1): continue
            for spec in rings_of_order(n):
                name = "x".join("%s%d" % s for s in spec)
                if name in done or len(spec) > int(os.environ.get("MAXCOMP", "9")): continue
                run_ring(spec, out, maxorb, budget)
