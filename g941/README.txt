g941/ -- the earlier certificates on induced subgraphs of G_941 (not needed for the results of the paper)
=========================================================================================================

Section 6 of the paper mentions that the method was first applied to induced subgraphs of the graph G_941 of
Dudek and Roedl (vertex set Z_941; two vertices are adjacent if their difference is a fifth power).  This folder
contains those certificates.  They give weaker bounds than the main result (483) and are kept for the record.

certificates/
  folkman_upper_753.txt    four cosets of the subgroup of order 47 removed: 753 vertices
  folkman_upper_705.txt    five cosets and the vertex 0 removed: 705 vertices
  folkman_upper_705b.txt   another choice of five cosets, and the vertex 0: 705 vertices
  folkman_upper_704.txt    the certificate lives on the graph of folkman_upper_705b.txt, and condition (ii) of
                           Lemma 3 is counted on this graph minus one further vertex: 704 vertices
verify_exact.py            the verification program for these files (exact integer arithmetic).  It works like
                           verify_q2.py, for graphs on Z_n with a multiplier as symmetry; the comment at its top
                           explains the format and the method.
logs/                      the output of verify_exact.py --blas-products for the four files (written by
                           run_all.sh of the main folder); each ends with THEOREM CHECKED
search/                    the programs and solver logs with which the certificates for 753 and 705 vertices were
                           found; the programs for 705b and 704 are in ../search/programs (ipm_b.py, drive_sub.py)

To check one file:   python3 verify_exact.py certificates/folkman_upper_704.txt --blas-products
(less than a minute on four cores; without the option the program uses integers only and takes a few minutes).
The program starts one process for each core and expects one BLAS thread in each process.  run_all.sh sets four
BLAS threads, which the other programs use; this slows the step down, and the logs in logs/ show about four minutes
for each file.
The comment lines of the certificate files name the program verify_spectral_cert.py; this was an earlier
floating-point checker, and verify_exact.py replaces it.
