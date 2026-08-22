"""
Script 5: Build the master benchmark dataset
Input: sabdab_post_cutoff_dedup.tsv, sequences.tsv, antigen_sequences.tsv, cdr_annotations.tsv
Output: benchmark_master.tsv

Merges all annotations into one clean file.
This is the primary input for all downstream analyses.

Final dataset: 304 antibody-antigen complexes
- Post Boltz-2 training cutoff (June 1 2023)
- X-ray crystallography, resolution <= 2.5 Angstrom
- Protein antigens only
- IMGT CDR annotations
- 171 conventional (VH+VL), 133 VHH nanobodies
"""

import pandas as pd
import ast

# Load all components
df_meta = pd.read_csv('sabdab_post_cutoff_dedup.tsv', sep='\t')
df_seqs = pd.read_csv('sequences.tsv', sep='\t')
df_ag = pd.read_csv('antigen_sequences.tsv', sep='\t')
df_cdr = pd.read_csv('cdr_annotations.tsv', sep='\t')

# Parse antigen sequences
df_ag['antigen_seqs_parsed'] = df_ag['antigen_seqs'].apply(ast.literal_eval)

# Keep only needed columns to avoid duplicate column conflicts
keep_meta = ['pdb', 'date', 'resolution', 'Hchain', 'Lchain',
             'antigen_chain', 'antigen_name', 'antigen_species']
keep_seqs = ['pdb', 'hseq', 'lseq', 'has_light']
keep_cdr = ['pdb', 'cdr_h1', 'cdr_h2', 'cdr_h3', 'cdr_h3_len']

master = df_meta[keep_meta].copy()
master = master.merge(df_seqs[keep_seqs], on='pdb', how='inner')
master = master.merge(df_cdr[keep_cdr], on='pdb', how='inner')

master.to_csv('benchmark_master.tsv', sep='\t', index=False)

print(f"Master benchmark set: {len(master)} complexes")
print(f"Columns: {list(master.columns)}")
print(f"\nVHH (no light chain): {master['lseq'].isna().sum()}")
print(f"Conventional (VH+VL): {master['lseq'].notna().sum()}")
print(f"\nCDR-H3 length distribution:")
print(master['cdr_h3_len'].describe())
