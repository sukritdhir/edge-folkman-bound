search/ -- how the graphs and the certificates were found (not needed for the proofs)
===================================================================================

This folder documents the search that Section 6 of the paper describes.  Nothing in it is used in a proof: a
certificate file is checked by verify_q2.py and check_independent.py without any reference to the way it was
found.

programs/   the programs of the search, as they were when the certificates were found.  They are working
            files: they were written for one machine, call each other, expect a folder scan/ for their output,
            and are not documented beyond their comments.  The main ones:
              gsym.py        builds the graphs Q(p,e), their induced subgraphs and the orbits of a group
              ipm_g.py       the interior-point solver for the semidefinite programme of Section 6 (with ipm_b.py)
              export_q.py    writes a certificate file from a result of the solver
              qfam.py, qscan.py, qspec.py   the survey of the family Q(p,e)
              normclass.py   unions of residue classes modulo 12 (the graph Y)
              vcost.py, pick_extra.py, pick_ls.py   the sets X of the certificates on 534 and 533 vertices
              splitclass.py, cells.py, orbsearch.py  orbits of the diagonal matrices of order 23 (the graph W_506)
              orbdens.py     the ranking of the unions of seven orbits by triangle density (the graph W)
            The other files belong to the earlier searches among Cayley graphs and to the work on G_941.

logs/       the output of the runs that Section 6 quotes.  In the solver logs, the lines that begin with CENTRAL
            give the value of beta after each stage; the last such line is the value quoted in the paper.
              t69_4_12_13_15_16_21_22.log   the graph W (483 vertices); the orbits are numbered 0..23 here
              t69_a_A0.log, t69_a_A.log     W_489 and W_495;  t69_a.log, t69_a_A1.log: the same seven orbits
                                            without the classes [(a,0)], and with the classes [(0,a)] only
              t69_4_5_12_13_16_21_22.log, t69_2_12_16_17_19_20_21.log, t69_1_5_6_16_17_19_21.log
                                            three other unions of seven orbits (they seem to fail)
              s277_506a.log                 W_506;  s277_2345.log: the union of 24 orbits (552 vertices);
                                            s277_483a.log: a union of 21 orbits (stopped above 1)
              y277_3.log, y277_3c.log       the graph Y;  y277_x23.log: the certificate on 533 vertices;
                                            y277_x24.log, y277_x26.log: sets of 24 and 26 vertices (above 1)
              y277_4.log, y97_11.log        two smaller subgraphs, of Q(277,6) on 417 vertices and of Q(97,6)
                                            on 539 vertices (both above 1)
              normclass.log, splitclass.log, splitclass157.log, orbsearch.log, qscan_43_300.log
                                            tables of the test with equal weights for families of subgraphs
              circ_search.log, gfq_dfs.log, noncyc_150_703.log, ring2_300_703.log, ring2_705_940.log,
              ring_sumfree.log, ringq_25_450.log, union_942_1300.log, union_dfs_comp.log
                                            the searches among Cayley graphs that preceded the family Q(p,e)

All numbers in these logs are floating-point values.
