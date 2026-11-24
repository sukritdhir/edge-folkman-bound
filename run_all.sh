#!/bin/bash
# run_all.sh -- repeats all computations of the paper and writes their output to logs/, tests/, graphs/ and g941/logs/.
# Start it in the folder that contains this file:   bash run_all.sh
# Needs Python 3 with NumPy and SciPy, 8 GB of memory, and about one hour on a computer with four cores.
# The folders logs/, tests/, graphs/ and g941/logs/ of the supplement contain the output of exactly this script.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p logs tests graphs g941/logs

python3 - > logs/environment.txt <<'EOF'
import sys, os, platform, numpy, scipy
print("python %s, numpy %s, scipy %s, %s %s, %d cores" % (sys.version.split()[0], numpy.__version__, scipy.__version__,
                                                      platform.system(), platform.machine(), os.cpu_count()))
EOF
cat logs/environment.txt

for f in certificates/folkman_upper_*.txt; do
  N=$(basename "$f" | sed 's/folkman_upper_\([0-9]*\)_.*/\1/')
  echo "== $f"
  # 1. the exact check, all-integer mode and fast mode (verify_q2.py itself uses one thread)
  python3 verify_q2.py "$f" --json logs/exact_$N.int.json > logs/exact_$N.int.log
  python3 verify_q2.py "$f" --blas-products --json logs/exact_$N.fast.json > logs/exact_$N.fast.log
  # 2. the independent check
  python3 check_independent.py "$f" --json logs/independent_$N.json > logs/independent_$N.log
  tail -n 2 logs/exact_$N.int.log
  tail -n 2 logs/independent_$N.log
done

# 3. Q(157,6) with equal weights (mentioned in the caption of Table 1; exact check only)
python3 verify_q2.py certificates/q157_6_uniform_948.txt --blas-products --json logs/exact_q157.fast.json > logs/exact_q157.fast.log
tail -n 2 logs/exact_q157.fast.log

# 4. the graphs of the paper from their definitions; the test with equal weights; Table 1
python3 paper_graphs.py certificates --json logs/paper_graphs.json > logs/paper_graphs.log
tail -n 1 logs/paper_graphs.log
python3 unweighted.py certificates/folkman_upper_*.txt > logs/unweighted.log
python3 qfamily_table.py > logs/qfamily_table.log

# 5. modified certificates and the negative control
MAIN=$(ls certificates/folkman_upper_*.txt | sort | head -n 1)
python3 make_tests.py "$MAIN" tests --run > logs/tests.log
tail -n 1 logs/tests.log

# 6. the two graphs of Theorem 2 as plain files
python3 export_graphs.py "$MAIN" graphs/W > logs/export_graphs.log
python3 export_graphs.py certificates/folkman_upper_588_q97.txt graphs/Q_97_6 >> logs/export_graphs.log
cat logs/export_graphs.log

# 7. the earlier certificates on subgraphs of G_941 (Section 6 of the paper; not needed for its results)
for f in g941/certificates/folkman_upper_*.txt; do
  N=$(basename "$f" .txt | sed 's/folkman_upper_//')
  python3 g941/verify_exact.py "$f" --blas-products --json g941/logs/exact_$N.json > g941/logs/exact_$N.log
  tail -n 2 g941/logs/exact_$N.log
done
echo "$SECONDS" > logs/run_all_seconds.txt
echo "ALL DONE in $SECONDS seconds"
