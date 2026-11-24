"""Run the interior-point solver on a queue of coset sets (vertex 0 removed), a few at a time.
usage: python3 exp/queue_run.py queuefile nworkers screen_beta
The queue file has one coset set per line (e.g. 0,6,11,12,16); lines can be appended while the driver runs.
Each run: OMP_NUM_THREADS=2 SCREEN=screen_beta python3 -u exp/ipm_sym.py SET 1 uniform scan/q_SET.pkl none 0.06 3e-6
Results (last central point of each run) are appended to queuefile + ".results"."""
import sys, os, subprocess, time, re
qf = sys.argv[1]; nw = int(sys.argv[2]); scr = sys.argv[3]
started = set(); running = {}
def summary(tag):
    cen = [l for l in open("scan/q_%s.log" % tag) if l.startswith("CENTRAL")]
    stop = [l.strip() for l in open("scan/q_%s.log" % tag) if "STOP" in l or l.startswith("finished")]
    if not cen: return "no central point"
    out = []
    for l in cen:
        m = re.match(r"CENTRAL t = ([0-9.e+]+): beta = ([0-9.]+)", l); out.append("%s:%s" % (m.group(1), m.group(2)))
    return " ".join(out[1:]) + "  | " + (stop[0][:60] if stop else "")
idle = 0
while True:
    for tag in list(running):
        if running[tag].poll() is not None:
            with open(qf + ".results", "a") as f: f.write("%s  %s\n" % (tag.replace("_", ","), summary(tag)))
            del running[tag]
    lines = [l.split()[0] for l in open(qf) if l.strip() and not l.startswith("#")]
    for s in lines:
        if len(running) >= nw: break
        tag = s.replace(",", "_")
        if tag in started: continue
        started.add(tag)
        if os.path.exists("scan/q_%s.log" % tag) and any(("STOP" in l or l.startswith("finished")) for l in open("scan/q_%s.log" % tag)):
            continue                                                  # done in an earlier driver run
        env = dict(os.environ, OMP_NUM_THREADS="2", SCREEN=scr)
        running[tag] = subprocess.Popen(["python3", "-u", "exp/ipm_sym.py", s, "1", "uniform", "scan/q_%s.pkl" % tag, "none", "0.06", "3e-6"],
                                        stdout=open("scan/q_%s.log" % tag, "w"), stderr=subprocess.STDOUT, env=env)
    if not running:
        idle += 1
        if idle > 720: break                                          # two hours without work: stop
    else: idle = 0
    time.sleep(10)
