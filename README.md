# The edge Folkman number F_e(3,3;4) is at most 483

This repository contains the paper

> Sukrit Dhir, *The edge Folkman number F_e(3,3;4) is at most 483*, November 2026

together with the certificate files, two programs that check them, the output of these programs, and a formal
proof of the main lemma for the proof assistant Lean 4.

The paper and the supplementary files are archived at Zenodo: https://doi.org/10.5281/zenodo.24491467. This
repository holds the same files, unpacked. The paper is a preprint and has not been peer reviewed.

The edge Folkman number F_e(3,3;4) is the smallest number of vertices of a graph that contains no K_4 and has a
monochromatic triangle in every colouring of its edges with two colours. The previous upper bound, 786, is due to
Lange, Radziszowski and Xu (2012). The paper proves the bound 483 with an induced subgraph W of the graph
Q(277,6), one of Mathon's antipodal distance-regular covers of complete graphs.

## Check the result yourself

The shortest check needs Python 3 with NumPy and takes less than a minute:

```
python3 verify_q2.py certificates/folkman_upper_483_q277s.txt
```

The program builds the graph, checks that it has no K_4, and checks the two hypotheses of Lemma 3 of the paper in
exact integer arithmetic. It ends with

```
THEOREM CHECKED: the graph G'' on 483 vertices is K4-free and every 2-colouring of its edges contains a
monochromatic triangle.  Hence F_e(3,3;4) <= 483.
```

A second program, written separately, checks the same certificate without using the symmetry of the graph
(about 2 minutes, needs NumPy and SciPy and 4.3 GB of memory):

```
python3 check_independent.py certificates/folkman_upper_483_q277s.txt
```

`bash run_all.sh` repeats all computations of the paper (47 minutes on our computer with four cores; allow one
hour and 8 GB of free memory). The folders `logs/`, `tests/`, `graphs/` and `g941/logs/` contain the output of
exactly this script.

## What is where

| Path | Content |
|---|---|
| `folkman-upper-bound.pdf` | the paper |
| `README.txt` | the full list of files |
| `certificates/` | the certificate files (the paper prints the SHA-256 checksums of the eight files `folkman_upper_*`) |
| `verify_q2.py` | the verification program (exact integer arithmetic) |
| `check_independent.py` | the second verification program |
| `logs/`, `tests/` | the output of both programs for all certificates and for modified certificates |
| `graphs/` | the graphs W and Q(97,6) as lists of edges and in the graph6 format |
| `lean/` | Lemma 3 for Lean 4 with Mathlib, and the output of Lean |
| `search/`, `g941/` | how the graphs and certificates were found, and earlier certificates (not needed for the proofs) |

## Use of AI

The results, the programs and the text of the paper were produced by an AI system (Claude, Anthropic) in a
research session set up and directed by the author. See the statement at the end of the paper.

## Licence

The paper and the files of this repository are made available under the Creative Commons Attribution 4.0
International licence (CC BY 4.0).
