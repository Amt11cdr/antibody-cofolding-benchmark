
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
