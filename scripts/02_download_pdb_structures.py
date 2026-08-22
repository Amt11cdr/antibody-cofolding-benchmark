"""
Script 2: Download experimental PDB structures from RCSB
Input: sabdab_post_cutoff_dedup.tsv
Output: pdb_structures/ directory with 304 PDB files

These are the experimental (native) structures used as ground truth
for DockQ and CDR-H3 RMSD computation.

Requirements: pip install pandas
"""

import pandas as pd
import urllib.request
import os
import time

df = pd.read_csv('sabdab_post_cutoff_dedup.tsv', sep='\t')
os.makedirs('pdb_structures', exist_ok=True)

pdb_ids = df['pdb'].tolist()
print(f"Downloading {len(pdb_ids)} PDB structures...")

failed = []
for i, pdb_id in enumerate(pdb_ids):
    outpath = f"pdb_structures/{pdb_id}.pdb"
    if os.path.exists(outpath):
        continue
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, outpath)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pdb_ids)} done...")
        time.sleep(0.1)
    except Exception as e:
        print(f"  FAILED: {pdb_id} — {e}")
        failed.append(pdb_id)

print(f"\nDone. Downloaded: {len(pdb_ids) - len(failed)}")
print(f"Failed: {len(failed)}")
if failed:
    print("Failed IDs:", failed)
