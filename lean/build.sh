#!/bin/bash
# build.sh -- lets Lean check FolkmanLemma/FolkmanLemma.lean and writes the result to build.log.
# Needs Lean 4.36.0 (`lean` and `lake` on the PATH, for example through elan) and an internet connection for the
# first run, which fetches Mathlib.  Without a Mathlib cache the first run compiles the needed part of Mathlib
# from source; this took about 30 minutes on four cores.
cd "$(dirname "$0")/FolkmanLemma" || exit 1
{
  echo "sha256 of FolkmanLemma.lean: $(sha256sum FolkmanLemma.lean | cut -d' ' -f1)"
  lean --version
  lake build 2>&1
  echo "lake build finished with status $?"
} > ../build.log
grep -v "designed for use with the module system" ../build.log | tail -n 6
