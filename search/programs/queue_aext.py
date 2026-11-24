"""Screen single-vertex extension certificates: for each listed vertex v run the interior-point solver on the base
graph (cosets, vertex 0 removed) with target = base - {v}, warm-started from init.pkl, and stop at the first central
point with t > 1e5 (SCREEN=0.5).  Results go to outprefix.results.
usage: python3 exp/queue_aext.py cosets init.pkl outprefix nworkers v1 v2 ..."""
import sys, os, subprocess, time, re
cosets, init, pref, nw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); vs = sys.argv[5:]
running = {}; todo = list(vs)
def summary(v):
    cen = [l for l in open("%s_v%s.log" % (pref, v)) if l.startswith("CENTRAL")]
    out = []
    for l in cen:
        m = re.match(r"CENTRAL t = ([0-9.e+]+): beta = ([0-9.]+)", l); out.append("%s:%s" % (m.group(1), m.group(2)))
    return " ".join(out)
while todo or running:
    for v in list(running):
        if running[v].poll() is not None:
            with open(pref + ".results", "a") as f: f.write("v %s  %s\n" % (v, summary(v)))
            del running[v]
    while todo and len(running) < nw:
        v = todo.pop(0)
        env = dict(os.environ, OMP_NUM_THREADS="2", SCREEN=os.environ.get("SCREEN", "0.5"))
        running[v] = subprocess.Popen(["python3", "-u", "exp/ipm_sym.py", cosets, "1", init, "%s_v%s.pkl" % (pref, v), "v:" + v, "0.003", "3e-6"],
                                      stdout=open("%s_v%s.log" % (pref, v), "w"), stderr=subprocess.STDOUT, env=env)
    time.sleep(5)
