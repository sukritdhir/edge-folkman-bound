"""Strict screening of sets of removed K'-cosets (vertex 0 removed as well) with exp/ipm_b.py (method A, TOLMID=1).

Each set is run from the uniform start along the central path up to t >= TSTOP (default 4.2e5) and its central values
are compared, at equal t, with the strict central path of the reference set 2,4,7,12,17 (optimum 0.99854):
    excess(t) = beta_set(t) - beta_ref(t)     (interpolated in log t).
The central values at equal t differ by about the same amount as the optima (checked on 0,3,7,12,18 and on the
706-vertex graph), so   estimated optimum = 0.99854 + excess at the last central point.
A run is stopped early when the excess is above STOPX (default 1.0e-3) at a central point with t >= 1e5.
usage: python3 exp/screen5.py queue.txt [rm0]       results are appended to queue.txt.results, logs scan/s5_<set>.log
"""
import os, sys, subprocess, re, time
import numpy as np
ref_t = np.array([6.913e4, 7.739e4, 8.767e4, 1.006e5, 1.171e5, 1.37e5, 1.623e5, 1.93e5, 2.322e5, 2.806e5, 3.435e5, 4.221e5, 5.27e5, 6.601e5, 8.43e5])
ref_b = np.array([1.00813899, 1.00615856, 1.00471347, 1.00314112, 1.00218011, 1.00121446, 1.00063833, 1.00006432, 0.99972653, 0.99938972, 0.99919623, 0.99900249, 0.99889468, 0.99878584, 0.99872755])
def ref(t): return float(np.interp(np.log(t), np.log(ref_t), ref_b))
qf = sys.argv[1]; rm0 = sys.argv[2] if len(sys.argv) > 2 else "1"
TSTOP = float(os.environ.get("TSTOP", "4.2e5")); STOPX = float(os.environ.get("STOPX", "1.0e-3"))
done = set()
if os.path.exists(qf + ".results"):
    done = set(l.split()[0] for l in open(qf + ".results") if l.strip())
for line in open(qf):
    S = line.strip()
    if not S or S.startswith("#") or S in done: continue
    tag = S.replace(",", "_"); log = "scan/s5_%s.log" % tag; t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="4", POOL="4", TOLMID="1", TSTOP=str(TSTOP))
    res = None
    for eps in ("0.06", "0.12"):
        f = open(log, "w")
        p = subprocess.Popen(["python3", "-u", "exp/ipm_b.py", S, rm0, "uniform", "scan/s5_%s.pkl" % tag, "none", eps, "3e-6"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        last = None; early = False; failed = False
        for l in p.stdout:
            f.write(l); f.flush()
            m = re.match(r"CENTRAL t = ([0-9.e+]+): beta = ([0-9.]+)", l)
            if m:
                t = float(m.group(1)); b = float(m.group(2)); last = (t, b, b - ref(t))
                if t >= 1e5 and b - ref(t) > STOPX:
                    early = True; p.terminate(); break
            if "not strictly feasible" in l: failed = True
        p.wait(); f.close()
        if last is not None or not failed: break
    if last is None: out = "%s  FAILED (no central point)  [%.0fs]" % (S, time.time() - t0)
    else: out = "%s  t %.4g  beta %.8f  excess %+.6f  estimated optimum %.5f  %s [%.0fs]" % (S, last[0], last[1], last[2], 0.99854 + last[2], "(stopped early)" if early else "", time.time() - t0)
    open(qf + ".results", "a").write(out + "\n"); print(out); sys.stdout.flush()
