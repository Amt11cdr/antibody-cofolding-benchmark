import os
"""
Script 21: CDR-H3 kink classification from experimental PDB structures
A kinked CDR-H3 has an aspartate at IMGT position H95 (or nearby)
creating a characteristic backbone geometry.
Classification based on presence of D/E at H95 and backbone torsion.

Input: pdb_structures/, benchmark_master.tsv
Output: cdr_h3_kink_classification.tsv
"""

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings
warnings.filterwarnings('ignore')

parser = PDBParser(QUIET=True)
df = pd.read_csv('benchmark_master.tsv', sep='\t')

results = []
failed = []

print(f"CDR-H3 kink classification for {len(df)} complexes...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    hchain = row['Hchain']
    cdr_h3_seq = row['cdr_h3']

    if not isinstance(cdr_h3_seq, str):
        failed.append(pdb_id)
        continue

    native_path = f'pdb_structures/{pdb_id}.pdb'
    if not os.path.exists(native_path):
        failed.append(pdb_id)
        continue

    try:
        structure = parser.get_structure(pdb_id, native_path)

        for model in structure:
            for chain in model:
                if chain.id != hchain:
                    continue

                residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
                full_seq = ''.join([seq1(r.resname, undef_code='X') for r in residues])
                idx = full_seq.find(cdr_h3_seq)

                if idx == -1:
                    failed.append(pdb_id)
                    break

                cdr_h3_residues = residues[idx:idx+len(cdr_h3_seq)]
                cdr_h3_len = len(cdr_h3_seq)

                # Kink classification:
                # 1. Check for D or E at position -4 from C-terminal end of CDR-H3
                # (IMGT H95 equivalent — the canonical kink-inducing residue)
                kink_position = max(0, cdr_h3_len - 4)
                kink_residue = seq1(cdr_h3_residues[kink_position].resname, undef_code='X')
                has_kink_residue = kink_residue in ['D', 'E']

                # 2. Check backbone geometry at kink position
                # Kinked CDR-H3 shows characteristic CA-CA distance pattern
                try:
                    ca_coords = np.array([
                        r['CA'].get_vector().get_array()
                        for r in cdr_h3_residues if 'CA' in r
                    ])

                    # Compute end-to-end distance ratio
                    if len(ca_coords) >= 4:
                        end_to_end = np.linalg.norm(ca_coords[-1] - ca_coords[0])
                        contour = sum(np.linalg.norm(ca_coords[j+1] - ca_coords[j])
                                     for j in range(len(ca_coords)-1))
                        extension_ratio = end_to_end / contour if contour > 0 else 0
                    else:
                        extension_ratio = np.nan

                except:
                    extension_ratio = np.nan

                # Kinked loops tend to be more compact (lower extension ratio)
                is_kinked = has_kink_residue and (extension_ratio < 0.5 if not np.isnan(extension_ratio) else has_kink_residue)

                results.append({
                    'pdb': pdb_id,
                    'cdr_h3_seq': cdr_h3_seq,
                    'cdr_h3_len': cdr_h3_len,
                    'kink_residue': kink_residue,
                    'has_kink_residue': has_kink_residue,
                    'extension_ratio': round(extension_ratio, 4) if not np.isnan(extension_ratio) else np.nan,
                    'is_kinked': is_kinked
                })
                break

    except Exception as e:
        failed.append(pdb_id)

import os
df_kink = pd.DataFrame(results)
df_kink.to_csv('cdr_h3_kink_classification.tsv', sep='\t', index=False)

print(f"\nDone. Classified: {len(results)}, Failed: {len(failed)}")
print(f"\nKink distribution:")
print(f"Kinked: {df_kink['is_kinked'].sum()} ({df_kink['is_kinked'].mean()*100:.1f}%)")
print(f"Non-kinked: {(~df_kink['is_kinked']).sum()} ({(~df_kink['is_kinked']).mean()*100:.1f}%)")
print(f"\nMean extension ratio by kink status:")
print(df_kink.groupby('is_kinked')['extension_ratio'].mean())
