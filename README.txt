Supplement to the paper
    Sukrit Dhir, "The edge Folkman number F_e(3,3;4) is at most 483"
Version 1, November 2026
==========================================================================================================

WHAT IS PROVED
The paper proves that a certain graph W on 483 vertices contains no K4 and that every colouring of its edges
with two colours contains a monochromatic triangle.  Hence F_e(3,3;4) <= 483.  The proof consists of a lemma
(Lemma 3 of the paper) and a computation that checks the hypotheses of the lemma for a certificate file.
This supplement contains the certificate files, two programs that check them, the output of these programs,
and a formal proof of the lemma for the proof assistant Lean.

THE SHORTEST CHECK (less than a minute; needs Python 3 and NumPy)
    python3 verify_q2.py certificates/folkman_upper_483_q277s.txt
The program builds the graph, checks that it has no K4, and checks the two hypotheses of the lemma in exact
integer arithmetic.  It ends with the lines
    THEOREM CHECKED: the graph G'' on 483 vertices is K4-free and every 2-colouring of its edges contains a
    monochromatic triangle.  Hence F_e(3,3;4) <= 483.
The comment at the top of verify_q2.py explains the format of a certificate file, the lemma with its proof, and
every step of the check.  The names differ slightly from the paper: the graphs G' and G of Lemma 3 are called
G' and G'' in the programs, and the numbers d_e are called D.

THE SECOND CHECK (about 2 minutes; needs NumPy and SciPy and 4.3 GB of memory)
    python3 check_independent.py certificates/folkman_upper_483_q277s.txt
This program was written separately and shares no code with the first.  It builds the graph from vectors, uses
no symmetry, and proves the positive definiteness of the whole matrix (21666 rows) by a floating-point Cholesky
factorisation with an explicit error bound.  It ends with "INDEPENDENT CHECK PASSED ...".  For the other
certificates it needs up to 5 minutes and up to 7.3 GB of memory.

EVERYTHING (47 minutes on our computer with four cores; allow one hour and 8 GB of free memory)
    bash run_all.sh
repeats all computations of the paper.  The folders logs/, tests/, graphs/ and g941/logs/ contain the output of
exactly this script; logs/environment.txt names the versions of Python, NumPy and SciPy that were used.

FILES
  README.txt               this file
  run_all.sh               repeats all computations (see above)

  certificates/            the certificate files (the paper prints the SHA-256 checksums of the eight
                           files folkman_upper_*)
    folkman_upper_483_q277s.txt   the graph W of Theorem 2(b): the certificate for F_e(3,3;4) <= 483
    folkman_upper_588_q97.txt     the graph Q(97,6) of Theorem 2(a), 588 vertices
    folkman_upper_489_q277s.txt, folkman_upper_495_q277s.txt, folkman_upper_506_q277s.txt,
    folkman_upper_533_q277.txt, folkman_upper_534_q277.txt, folkman_upper_556_q277.txt
                                  the further certificates of Section 4.3 (not needed for the main result)
    q157_6_uniform_948.txt        the graph Q(157,6) with equal weights (caption of Table 1)
    The comment lines of some certificate files name the programs verify_q.py or verify_spectral_cert.py.
    These are earlier versions; verify_q2.py reads all files of this folder.

  verify_q2.py             the verification program of Section 5.1 (exact integer arithmetic)
  check_independent.py     the second program of Section 5.2
  paper_graphs.py          builds every graph from the definition printed in the paper and compares it with
                           the graph of the certificate file; computes the numbers of Proposition 13
  unweighted.py            the numbers beta_0 (the test with equal weights) of Table 2 and Section 6
  qfamily_table.py         Table 1
  make_tests.py            writes modified certificates and a negative control and runs both programs on them
  export_graphs.py         writes a graph as a list of edges and in the graph6 format

  logs/                    output of run_all.sh
    exact_N.int.log, .json        verify_q2.py for the certificate on N vertices, all-integer mode
    exact_N.fast.log, .json       the same with the option --blas-products
    independent_N.log, .json      check_independent.py
    exact_q157.fast.log, .json    verify_q2.py for Q(157,6)
    paper_graphs.log, .json; unweighted.log; qfamily_table.log; tests.log; export_graphs.log
    environment.txt, run_all_seconds.txt
  tests/                   the modified certificates, the output of both programs for each of them, and the
                           summary results.txt (both programs accept the one valid file and reject the others)
  graphs/                  W and Q(97,6) as plain files: X.edges.txt (first line "n m", then one edge per line),
                           X.g6 (graph6) and X.vertices.txt (the vector that belongs to each vertex)

  lean/                    Lemma 3 for the proof assistant Lean 4 with Mathlib, and the output of Lean
                           (see lean/README.txt)

  search/                  NOT NEEDED FOR THE PROOFS: the programs that were used to find the graphs and the
                           certificates, and the logs of the solver runs that Section 6 quotes (see search/README.txt)
  g941/                    NOT NEEDED FOR THE RESULTS OF THE PAPER: the earlier certificates on induced subgraphs of
                           the graph G_941 (753, 705 and 704 vertices) that Section 6 mentions, with their own
                           verification program (see g941/README.txt)

HOW THE PARTS DEPEND ON EACH OTHER
The proof of the main theorem needs only: Lemma 3 (paper, Section 2; formal proof in lean/), the file
certificates/folkman_upper_483_q277s.txt, and one run of verify_q2.py.  Everything else is a cross-check or
documentation.

USE OF AI
The results, the programs and the text of the paper were produced by an AI system (Claude, Anthropic) in a
research session set up and directed by the author; see the statement at the end of the paper.
