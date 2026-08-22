"""
Script 3: Extract antibody and antigen sequences from PDB structures
Input: sabdab_post_cutoff_dedup.tsv, pdb_structures/
Output: sequences.tsv, antigen_sequences.tsv

Extracts:
- Heavy chain sequence (hseq)
- Light chain sequence (lseq) where present
- Antigen chain sequence(s) per complex
"""

import pandas as pd
import os
import ast
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings
warnings.filterwarnings('ignore')

parser = PDBParser(QUIET=True)
df = pd.read_csv('sabdab_post_cutoff_dedup.tsv', sep='\t')

def extract_chain_sequence(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
                return ''.join([seq1(r.resname, undef_code='X') for r in residues])
    return None

# Extract antibody sequences
ab_results = []
ag_results = []
failed = []

for _, row in df.iterrows():
    pdb_id = row['pdb']
    pdb_path = f"pdb_structures/{pdb_id}.pdb"

    if not os.path.exists(pdb_path):
        failed.append(pdb_id)
        continue

    try:
        structure = parser.get_structure(pdb_id, pdb_path)

        hseq = extract_chain_sequence(structure, row['Hchain'])
        lseq = extract_chain_sequence(structure, row['Lchain']) if pd.notna(row['Lchain']) else None

        ab_results.append({
            'pdb': pdb_id,
            'hchain': row['Hchain'],
            'lchain': row['Lchain'],
            'hseq': hseq,
            'lseq': lseq,
            'has_light': pd.notna(row['Lchain'])
        })

        # Extract antigen sequences
        antigen_chains = str(row['antigen_chain']).split('|')
        antigen_seqs = {}
        for ag_chain in antigen_chains:
            ag_chain = ag_chain.strip()
            seq = extract_chain_sequence(structure, ag_chain)
            if seq:
                antigen_seqs[ag_chain] = seq

        ag_results.append({
            'pdb': pdb_id,
            'antigen_chain': row['antigen_chain'],
            'antigen_seqs': str(antigen_seqs)
        })

    except Exception as e:
        failed.append(pdb_id)
        print(f"Failed {pdb_id}: {e}")

pd.DataFrame(ab_results).to_csv('sequences.tsv', sep='\t', index=False)
pd.DataFrame(ag_results).to_csv('antigen_sequences.tsv', sep='\t', index=False)

print(f"Extracted: {len(ab_results)} complexes")
print(f"Failed: {len(failed)}")
