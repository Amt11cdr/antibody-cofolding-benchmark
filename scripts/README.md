# Antibody Co-Folding Benchmark — Analysis Scripts

## Project
Benchmarking Boltz-2 confidence calibration at the CDR-H3/antigen interface.
304 post-training-cutoff antibody-antigen complexes from SAbDab.
1,520 predictions (5 seeds x 304 complexes).

## Author
Amritansh Tiwari — MSc AI for Medicine, University College Dublin — August 2026

## Pipeline Order
01_download_sabdab.py — Download and filter SAbDab
02_download_pdb_structures.py — Download experimental PDB structures
03_extract_sequences.py — Extract antibody and antigen sequences
04_annotate_cdrs.py — Annotate CDR regions with ANARCI/IMGT
05_build_benchmark_master.py — Build master benchmark dataset
06_pae_submatrix_analysis.py — PAE submatrix residue imbalance analysis
07_cdr_h3_rmsd.py — CDR-H3 RMSD batch computation
08_seed_variance_analysis.py — Seed variance analysis
09_contact_position_extraction.py — Contact position extraction
10_contact_pae_analysis.py — Contact position PAE analysis
11_parse_dockq.py — DockQ parsing and confidence score merge

## Requirements
pip install pandas numpy scipy biopython gemmi anarci
brew install hmmer rclone

## Key Findings
ECE = 0.32 at CDR-H3/antigen interface
FPR = 44.6% at ipTM >= 0.7
CDR-H3 PAE < framework PAE in 74.7% of complexes (11.4x residue imbalance)
CDR-H3 RMSD varies 6.9x more than ipTM across seeds
Contact position PAE barely differs from non-contact PAE (ratio 1.06)

## Preprint
Coming soon — bioRxiv
