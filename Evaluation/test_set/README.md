# Test Set

This directory contains the compact, repository-friendly form of the FineDroid evaluation set.

## Files

- `buggy_traces_withGT.json`: 51 source traces with known NCF bugs and ground-truth information.
- `bug_free_traces_withGT.json`: 51 corresponding bug-free traces.
- `labels_ground_truth_mod.csv`: ground-truth bug descriptions indexed by case ID.

## Evaluation convention

**The released evaluation excludes the legacy `tr_33` and `fl_33` cases.**. After this exclusion, the final balanced test set contains 100 traces: 50 buggy and 50 bug-free.

All case-level outputs in the parent [`Evaluation`](../readme.md) directory follow this convention.

## Complete raw artifacts

Download `testingset.zip` from the [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing) for the corresponding APKs, execution screenshots, XML hierarchy files, and raw traces.

The files in this directory are the canonical compact copy for the GitHub artifact. The duplicate under the ignored local `testing/testset/` directory is not required for publication.
