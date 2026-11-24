"""Local search (simulated annealing) over K4-free circulants Cay(Z_n, S), S a union of orbits of a SMALL multiplier
group G = <s, -1> (many orbits; the depth-first scan union_scan.py only covered groups with at most 24 orbits).
Objective: the unweighted normalised ratio  lambda_min(Deg^(-1/2) M Deg^(-1/2)) / (-2/3)   (below 1 = the graph arrows).
usage: python3 exp/circ_search.py n gord [moves] [seed] [T0] [T1]
   gord = order of the multiplier s (the group is <s, -1>); gord = 1 means G = {+1, -1}.
Output: one line per improvement, final best in scan/circ_search.log (appended).
"""
import numpy as np, sys, time, random
from math import gcd


def mult_order(s, n):
    x = s % n; k = 1
    while x != 1:
        x = (x * s) % n; k += 1
        if k > n: return None
    return k


def find_group(n, gord):
    """orbits of <s, -1> on Z_n minus 0, for an element s of multiplicative order gord"""
    if gord == 1: s = 1
    else:
        s = None
        for a in range(2, n):
            if gcd(a, n) == 1 and mult_order(a, n) == gord: s = a; break
        if s is None: return None
    G = set(); x = 1
    while True:
        G.add(x); G.add(n - x); x = (x * s) % n
        if x == 1: break
    seen = [False] * n; orbs = []
    for x in range(1, n):
        if seen[x]: continue
        o = sorted(set((a * x) % n for a in G))
        for y in o: seen[y] = True
        orbs.append(o)
    return s, sorted(G), orbs


class Circ:
    def __init__(self, n, orbs):
        self.n = n; self.orbs = orbs; self.mask = (1 << n) - 1
        self.obits = [sum(1 << x for x in o) for o in orbs]; self.reps = [o[0] for o in orbs]
        self.freqs = np.array([0] + self.reps)

    def rot(self, x, c):
        n = self.n
        return ((x << c) | (x >> (n - c))) & self.mask

    def k4_at(self, Sb, c):
        W = Sb & self.rot(Sb, c); w = W
        while w:
            v = (w & -w).bit_length() - 1
            if W & self.rot(Sb, v): return True
            w &= w - 1
        return False

    def ratio(self, chosen):
        n = self.n
        S = np.array(sorted(x for i in chosen for x in self.orbs[i]), dtype=np.int64)
        if len(S) == 0: return 9.0, 0, 0, 0
        inS = np.zeros(n, dtype=bool); inS[S] = True
        Cp = S[2 * S < n]; m = len(Cp); idx = np.full(n, -1, dtype=np.int64); idx[Cp] = np.arange(m); idx[(n - Cp) % n] = np.arange(m)
        # triangles on the edge {0, c}: third vertex v in S with v - c in S
        T = inS[(S[None, :] - Cp[:, None]) % n]                    # (m, d)
        deg = T.sum(axis=1)
        if (deg == 0).any(): return 9.0, len(S), 0, 0
        jj, vv = np.nonzero(T); v = S[vv]; c = Cp[jj]
        # edge {0, v}: class idx[v], base 0 if 2v < n else v
        j2a = idx[v]; x2a = np.where(2 * v < n, 0, v)
        w = (v - c) % n
        j2b = idx[w]; x2b = np.where(2 * w < n, c, v)
        J = np.concatenate([jj, jj]); J2 = np.concatenate([j2a, j2b]); X2 = np.concatenate([x2a, x2b])
        flat = J * m + J2
        dm = 1.0 / np.sqrt(deg)
        ph = np.exp(2j * np.pi * ((self.freqs[:, None] * X2[None, :]) % n) / n)
        B = np.empty((len(self.freqs), m, m), dtype=complex)
        for f in range(len(self.freqs)):
            B[f] = (np.bincount(flat, weights=ph[f].real, minlength=m * m) + 1j * np.bincount(flat, weights=ph[f].imag, minlength=m * m)).reshape(m, m)
        B *= dm[None, :, None]; B *= dm[None, None, :]
        B = (B + B.conj().transpose(0, 2, 1)) / 2
        lam = np.linalg.eigvalsh(B)[:, 0].min()
        return float(lam / (-2 / 3)), len(S), int(deg.min()), int(deg.max())


def anneal(n, gord, moves=20000, seed=1, T0=0.03, T1=0.002, start=None, verbose=True, tmax=1e9):
    fg = find_group(n, gord)
    if fg is None: return None
    s, G, orbs = fg
    C = Circ(n, orbs); N = len(orbs); rng = random.Random(seed)
    chosen = set(start) if start else set(); Sb = 0
    for i in chosen: Sb |= C.obits[i]
    # greedy random start: add orbits while K4-free
    order = list(range(N)); rng.shuffle(order)
    for i in order:
        if i in chosen: continue
        S2 = Sb | C.obits[i]
        if not C.k4_at(S2, C.reps[i]): chosen.add(i); Sb = S2
    cur = C.ratio(chosen); best = (cur, sorted(chosen)); t0 = time.time()
    if verbose: print("n %d  |G| %d (s = %d)  orbits %d  start: ratio %.4f d %d tri/edge %d..%d" % (n, len(G), s, N, cur[0], cur[1], cur[2], cur[3])); sys.stdout.flush()
    for it in range(moves):
        if time.time() - t0 > tmax: break
        T = T0 * (T1 / T0) ** (it / max(1, moves - 1))
        r = rng.random()
        out = [i for i in range(N) if i not in chosen]
        if r < 0.25 and out:                                          # add
            i = rng.choice(out); S2 = Sb | C.obits[i]
            if C.k4_at(S2, C.reps[i]): continue
            new = chosen | {i}
        elif r < 0.40 and len(chosen) > 1:                            # remove
            i = rng.choice(sorted(chosen)); new = chosen - {i}; S2 = Sb & ~C.obits[i]
        elif out and chosen:                                          # swap
            i = rng.choice(sorted(chosen)); j = rng.choice(out)
            S2 = (Sb & ~C.obits[i]) | C.obits[j]
            if C.k4_at(S2, C.reps[j]): continue
            new = (chosen - {i}) | {j}
        else: continue
        val = C.ratio(new)
        if val[0] <= cur[0] or rng.random() < np.exp(-(val[0] - cur[0]) / T):
            chosen = new; Sb = S2; cur = val
            if cur[0] < best[0][0] - 1e-9:
                best = (cur, sorted(chosen))
                if verbose: print("  it %6d T %.4f  ratio %.4f  d %d  tri/edge %d..%d  [%.0fs]" % (it, T, cur[0], cur[1], cur[2], cur[3], time.time() - t0)); sys.stdout.flush()
    return best, s, len(G), [orbs[i][0] for i in best[1]], time.time() - t0


if __name__ == "__main__":
    n = int(sys.argv[1]); gord = int(sys.argv[2])
    moves = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    T0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
    T1 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.002
    res = anneal(n, gord, moves, seed, T0, T1)
    if res is None: print("no element of order", gord); sys.exit(0)
    best, s, go, reps, dt = res
    line = "n %d s %d |G| %d seed %d moves %d: best ratio %.4f d %d tri/edge %d..%d  orbit reps %s  [%.0fs]" % (n, s, go, seed, moves, best[0][0], best[0][1], best[0][2], best[0][3], reps, dt)
    print(line); open("scan/circ_search.log", "a").write(line + "\n")
