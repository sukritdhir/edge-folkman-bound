lean/ -- Lemma 3 of the paper (the weighted criterion) for the proof assistant Lean 4
====================================================================================

FolkmanLemma/FolkmanLemma.lean   the statement and the proof (about 170 lines; the comment at its top explains
                                 how the statement of the paper is written in Lean)
FolkmanLemma/lakefile.toml       the project file; it asks for Mathlib, release v4.36.0
FolkmanLemma/lean-toolchain      the version of Lean: leanprover/lean4:v4.36.0
FolkmanLemma/lake-manifest.json  the exact versions of Mathlib and of the packages it needs
build.sh                         runs the check and writes build.log
build.log                        the output of build.sh for the file in this folder

What was checked.  Lean accepted the theorem `Folkman.weighted_criterion`.  The file contains no `sorry`, and the
last line of the file asks Lean for the axioms on which the proof depends; the answer in build.log is
    'Folkman.weighted_criterion' depends on axioms: [propext, choice, Quot.sound]
These are the three standard axioms of Lean with classical logic.  The other lines of build.log are warnings
that concern the way Mathlib is imported; they do not concern the proof.

What was not checked.  The formal proof covers the lemma only.  The computations of the paper (that the graphs
contain no K4, the inequality (ii), and the positive definiteness of the matrices) are done by the programs
verify_q2.py and check_independent.py, not by Lean.

How to repeat the check.  Install Lean through elan (https://lean-lang.org), then run
    bash build.sh
in this folder.  The first run fetches Mathlib.  `lake exe cache get` inside FolkmanLemma/ downloads compiled
files of Mathlib; without it the needed part of Mathlib is compiled from source, which took about 30 minutes on
four cores.  The check of FolkmanLemma.lean itself takes a few seconds.
