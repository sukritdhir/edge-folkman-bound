#!/usr/bin/env python3
"""
make_tests.py -- writes modified certificates and a negative control, and (with --run) runs both verification
programs on them.  Not part of any proof: the purpose is to see that the programs do not accept wrong input.

Usage:  python3 make_tests.py CERTIFICATE OUTDIR [--run] [--down X] [--ok Y]

CERTIFICATE is the certificate of the main theorem (a file with the lines `matrix` and `D`).  The following
files are written to OUTDIR:

  t1_d_times_1.001.txt     all numbers D multiplied by 1.001      -> inequality (ii) fails
  t2_d_scaled_down.txt     all numbers D multiplied by 1 - X      -> (ii) holds, the matrix is not positive definite
  t3_d_scaled_ok.txt       all numbers D multiplied by 1 - Y      -> still a valid certificate (smaller slack)
  t4_extra_vertex.txt      a line `extra v` is added: one vertex is removed from the graph, the weights stay
                           -> (ii) fails
  t5_weight_zero.txt       the first positive weight in the file is set to 0   -> (ii) fails
  t6_wrong_matrix.txt      the last entry of the line `matrix` is increased by 1 -> the lists are not closed under
                           the permutation any more (one program reports that pi is not an automorphism, the other
                           that an image of a listed triple is not a triangle)
  n1_q101_5_uniform.txt    the graph Q(101,5), 510 vertices, equal weights    -> the matrix is not positive definite

Multiplying all D by 1 - x changes the smallest eigenvalue lambda of the normalised matrix to
(lambda - x) / (1 - x).  The defaults X = 0.003 and Y = 0.001 lie on the two sides of lambda, which is about
0.0018 for the certificate on 483 vertices.

With --run the programs verify_q2.py and check_independent.py (expected in the directory of this script) are
started on every file; the last line of each output is collected in OUTDIR/results.txt, and a sentence for the
paper is written to OUTDIR/tamper.tex.
"""
import sys, os, subprocess
from fractions import Fraction


def dec(x, digits):
    """Decimal string of the Fraction x, rounded down to `digits` digits after the point."""
    s = x.numerator * 10 ** digits // x.denominator
    return "%d.%0*d" % (s // 10 ** digits, digits, s % 10 ** digits)


def scaled(lines, factor):
    out = []
    for ln in lines:
        s = ln.split()
        if s and s[0] == "D":
            digits = len(s[3].split(".")[1]) if "." in s[3] else 0
            out.append("D %s %s %s\n" % (s[1], s[2], dec(Fraction(s[3]) * factor, digits)))
        else:
            out.append(ln)
    return out


def find_theta(p, e, nr):
    """Smallest (a, b), b > 0, such that the classes of (a + b t)^i, i < (p+1)e, are pairwise different."""
    H = sorted(set(pow(x, e, p) for x in range(1, p)))
    n = (p + 1) * e
    for b in range(1, p):
        for a in range(0, p):
            seen = set(); x0, x1 = 1, 0; ok = True
            for i in range(n):
                key = min(((h * x0) % p) * p + (h * x1) % p for h in H)
                if key in seen:
                    ok = False; break
                seen.add(key)
                x0, x1 = (a * x0 + nr * b * x1) % p, (b * x0 + a * x1) % p
            if ok:
                return a, b
    raise SystemExit("no generator found")


def main():
    args = sys.argv[1:]
    cert = args[0]; outdir = args[1]; run = "--run" in args
    X = Fraction(args[args.index("--down") + 1]) if "--down" in args else Fraction(3, 1000)
    Y = Fraction(args[args.index("--ok") + 1]) if "--ok" in args else Fraction(1, 1000)
    os.makedirs(outdir, exist_ok=True)
    lines = open(cert).readlines()
    files = []

    def write(name, content, expect):
        with open(os.path.join(outdir, name), "w") as fh:
            fh.writelines(content)
        files.append((name, expect))

    write("t1_d_times_1.001.txt", scaled(lines, Fraction(1001, 1000)), "reject")
    write("t2_d_scaled_down.txt", scaled(lines, 1 - X), "reject")
    write("t3_d_scaled_ok.txt", scaled(lines, 1 - Y), "accept")
    # one vertex less: the smallest vertex of the graph
    removed = set(); p = e0 = None
    for ln in lines:
        s = ln.split()
        if s and s[0] == "removed": removed.update(int(x) for x in s[1:])
        if s and s[0] == "p": p = int(s[1])
        if s and s[0] == "e0": e0 = int(s[1])
        assert not (s and s[0] == "extra"), "this script expects a certificate without a line `extra`"
    new = min(v for v in range((p + 1) * e0) if v not in removed)
    out = []
    for ln in lines:
        out.append(ln)
        if ln.startswith("matrix"):
            out.append("extra %d\n" % new)
    write("t4_extra_vertex.txt", out, "reject")
    out = []; done = False
    for ln in lines:
        s = ln.split()
        if not done and len(s) == 4 and s[0].isdigit() and Fraction(s[3]) > 0:
            out.append("%s %s %s 0\n" % (s[0], s[1], s[2])); done = True
        else:
            out.append(ln)
    write("t5_weight_zero.txt", out, "reject")
    out = []
    for ln in lines:
        s = ln.split()
        if s and s[0] == "matrix":
            out.append("matrix %s %s %s %d\n" % (s[1], s[2], s[3], int(s[4]) + 1))
        else:
            out.append(ln)
    write("t6_wrong_matrix.txt", out, "reject")
    a, b = find_theta(101, 5, 2)
    write("n1_q101_5_uniform.txt",
          ["# Negative control: the graph Q(101, 5) on 510 vertices with equal weights.  The test fails by 1.3 per cent.\n",
           "family Q\n", "p 101\n", "nr 2\n", "theta %d %d\n" % (a, b), "e0 5\n", "classes 0\n", "step 6\n", "uniform\n"],
          "reject")
    print("wrote %d files to %s" % (len(files), outdir))
    if not run:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    res = []; good = True
    for name, expect in files:
        path = os.path.join(outdir, name)
        for prog, extra_args in (("verify_q2.py", ["--blas-products"]), ("check_independent.py", [])):
            log = os.path.join(outdir, name[:-4] + "." + prog[:-3] + ".log")
            with open(log, "w") as fh:
                rc = subprocess.call([sys.executable, os.path.join(here, prog), path] + extra_args, stdout=fh, stderr=subprocess.STDOUT)
            last = [ln.strip() for ln in open(log) if ln.strip()][-1]
            verdict = "accept" if rc == 0 else "reject"
            good = good and verdict == expect
            res.append("%-26s %-22s expected %-6s got %-6s %s   | %s" % (name, prog, expect, verdict, "ok" if verdict == expect else "MISMATCH", last[:150]))
            print(res[-1], flush=True)
    with open(os.path.join(outdir, "results.txt"), "w") as fh:
        fh.write("\n".join(res) + "\n")
    with open(os.path.join(outdir, "tamper.tex"), "w") as fh:
        if good:
            fh.write("From the certificate of Proposition~\\ref{prop:cW} we made six modified files: all numbers $d_e$ multiplied by "
                     "$1.001$ (then (ii) fails), by $%s$ (then (ii) holds, but the matrix is not positive definite) and by $%s$ "
                     "(still a valid certificate, with a smaller slack); one vertex removed from the graph while the weights stay "
                     "((ii) fails); one weight set to $0$ ((ii) fails); and a wrong matrix for $\\pi$. Both programs accept the "
                     "third file and reject the other five.\n" % (float(1 - X), float(1 - Y)))
        else:
            fh.write("\\textbf{[THE TESTS DID NOT GIVE THE EXPECTED RESULTS: see tests/results.txt]}\n")
    print("all results as expected" if good else "MISMATCH: see results.txt")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
