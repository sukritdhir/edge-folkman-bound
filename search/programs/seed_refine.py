"""Refine a known K4-free circulant in the lattice of finer orbits: starting from S0 (union of orbits of a big multiplier
group), try to ADD orbits of a smaller group G = <s, -1> (gord = order of s; 1 = plus/minus pairs) keeping K4-free, and
report the ratio after each greedy addition (the addition with the best ratio is taken).
usage: python3 exp/seed_refine.py n hord rep1,rep2,.. gord [maxadd]      (S0 = union of the orbits of the order-hord group through the reps)"""
import sys, time
sys.path.insert(0, "exp"); sys.path.insert(0, ".")
from circ_search import find_group, Circ
from union_scan import sym_orbits, cyclic_subgroups_all
n = int(sys.argv[1]); hord = int(sys.argv[2]); reps = [int(x) for x in sys.argv[3].split(",")]; gord = int(sys.argv[4])
maxadd = int(sys.argv[5]) if len(sys.argv) > 5 else 10
H = [h for h in cyclic_subgroups_all(n) if len(h) == hord][0]
S0 = set(x for o in sym_orbits(n, H) if any(r in o for r in reps) for x in o)
s, G, orbs = find_group(n, gord); C = Circ(n, orbs)
chosen = set(i for i, o in enumerate(orbs) if o[0] in S0)
assert all(x in S0 for i in chosen for x in orbs[i]) and sum(len(orbs[i]) for i in chosen) == len(S0), "S0 is not a union of the fine orbits"
Sb = sum(C.obits[i] for i in chosen)
print("n %d: S0 d %d, fine group |G| %d, %d orbits; start ratio %s" % (n, len(S0), len(G), len(orbs), C.ratio(chosen)))
for step in range(maxadd):
    cand = []
    for i in range(len(orbs)):
        if i in chosen: continue
        S2 = Sb | C.obits[i]
        if C.k4_at(S2, C.reps[i]): continue
        cand.append((C.ratio(chosen | {i}), i))
    print("step %d: %d K4-free additions" % (step, len(cand)), end="")
    if not cand: print(); break
    cand.sort(); val, i = cand[0]
    print("; best: add orbit of %d -> ratio %.4f d %d tri/edge %d..%d   (worst %.4f)" % (orbs[i][0], val[0], val[1], val[2], val[3], cand[-1][0][0]))
    chosen.add(i); Sb |= C.obits[i]
print("final:", C.ratio(chosen), "reps added:", sorted(orbs[i][0] for i in chosen if orbs[i][0] not in S0))
