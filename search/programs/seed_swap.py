"""Neighbourhood search around a known K4-free circulant S0 in the lattice of plus/minus pairs (or orbits of a small
group): remove r orbits (all choices for r = 1, random samples for r = 2, 3), then add orbits greedily (each time the
K4-free addition with the best ratio) and report improvements of the unweighted ratio.
usage: python3 exp/seed_swap.py n hord rep1,rep2,.. gord [samples2] [samples3]"""
import sys, time, random, itertools
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
from circ_search import find_group, Circ
from union_scan import sym_orbits, cyclic_subgroups_all
n = int(sys.argv[1]); hord = int(sys.argv[2]); reps = [int(x) for x in sys.argv[3].split(",")]; gord = int(sys.argv[4])
ns2 = int(sys.argv[5]) if len(sys.argv) > 5 else 300; ns3 = int(sys.argv[6]) if len(sys.argv) > 6 else 300
H = [h for h in cyclic_subgroups_all(n) if len(h) == hord][0]
S0 = set(x for o in sym_orbits(n, H) if any(r in o for r in reps) for x in o)
s, G, orbs = find_group(n, gord); C = Circ(n, orbs); N = len(orbs)
base = set(i for i, o in enumerate(orbs) if o[0] in S0)
r0 = C.ratio(base); print("n %d: S0 d %d ratio %.4f; fine orbits %d (size %d)" % (n, len(S0), r0[0], N, len(G))); sys.stdout.flush()
best = (r0[0], sorted(base)); rng = random.Random(1); t0 = time.time()
def refill(chosen):
    chosen = set(chosen); Sb = sum(C.obits[i] for i in chosen); cur = C.ratio(chosen)
    while True:
        cand = []
        for i in range(N):
            if i in chosen: continue
            S2 = Sb | C.obits[i]
            if C.k4_at(S2, C.reps[i]): continue
            cand.append((C.ratio(chosen | {i}), i))
        if not cand: break
        cand.sort(); val, i = cand[0]
        chosen.add(i); Sb |= C.obits[i]; cur = val
    return cur, chosen
def trial(rem, tag):
    global best
    cur, ch = refill(base - set(rem))
    new = sorted(orbs[i][0] for i in ch - base)
    if set(ch) != base and cur[0] < best[0] - 1e-9:
        best = (cur[0], sorted(ch))
        print("  %s remove %s add %s -> ratio %.4f d %d tri/edge %d..%d  [%.0fs]" % (tag, [orbs[i][0] for i in rem], new, cur[0], cur[1], cur[2], cur[3], time.time() - t0)); sys.stdout.flush()
    return cur, ch
bl = sorted(base); cnt = 0; diff = 0
for i in bl:
    cur, ch = trial([i], "r=1"); cnt += 1; diff += (set(ch) != base)
print("r = 1: %d removals, %d lead to a different maximal set; best ratio so far %.4f  [%.0fs]" % (cnt, diff, best[0], time.time() - t0)); sys.stdout.flush()
for _ in range(ns2): trial(rng.sample(bl, 2), "r=2")
print("r = 2: best ratio so far %.4f  [%.0fs]" % (best[0], time.time() - t0)); sys.stdout.flush()
for _ in range(ns3): trial(rng.sample(bl, 3), "r=3")
print("final best ratio %.4f, reps %s" % (best[0], [orbs[i][0] for i in best[1]]))
