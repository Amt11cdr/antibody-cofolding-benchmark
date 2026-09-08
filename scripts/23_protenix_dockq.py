"""
Script 23: Protenix DockQ computation
Input: benchmark_master.tsv, Protenix CIF files on Drive
Output: protenix_dockq_results.tsv

Protenix CIF naming:
gdrive:antibody_benchmark/protenix_outputs/{pdb_id}/seed_{seed_num}/predictions/{pdb_id}_sample_0.cif

5 seeds per complex: 101, 102, 103, 104, 105
"""

import subprocess
import json
import os
import pandas as pd
import gemmi
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

DRIVE_PROT = 'gdrive:antibody_benchmark/protenix_outputs'
TMP = '/tmp/protenix_dockq'
OUTFILE = 'protenix_dockq_results.tsv'
SEEDS = [101, 102, 103, 104, 105]
os.makedirs(TMP, exist_ok=True)

df = pd.read_csv('benchmark_master.tsv', sep='\t')

# Load checkpoint
if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(zip(df_existing['pdb'], df_existing['seed']))
    results = df_existing.to_dict('records')
    print(f"Resuming: {len(results)} done")
else:
    done = set()
    results = []

failed = []

print(f"Starting Protenix DockQ for {len(df)} complexes...")

for idx, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    native_path = f'pdb_structures/{pdb_id}.pdb'

    if not os.path.exists(native_path):
        continue

    for seed in SEEDS:
        if (pdb_id, seed) in done:
            continue

        cif_name = f'{pdb_id}_sample_0.cif'
        cif_src = f'{DRIVE_PROT}/{pdb_id}/seed_{seed}/{cif_name}'
        cif_dst = f'{TMP}/{pdb_id}_seed{seed}.cif'
        pdb_dst = f'{TMP}/{pdb_id}_seed{seed}.pdb'

        try:
            if not os.path.exists(cif_dst):
                r = subprocess.run([
                    'rclone', 'copy', cif_src, TMP
                ], capture_output=True, timeout=120)
                src = f'{TMP}/{cif_name}'
                if os.path.exists(src):
                    os.rename(src, cif_dst)

            if not os.path.exists(cif_dst):
                failed.append((pdb_id, seed))
                continue

            # Convert CIF to PDB
            doc = gemmi.cif.read(cif_dst)
            gemmi.make_structure_from_block(doc.sole_block()).write_pdb(pdb_dst)

            # Run DockQ
            dockq_out = f'{TMP}/dq_{pdb_id}_seed{seed}.json'
            result = subprocess.run([
                'DockQ', pdb_dst, native_path, '--json', dockq_out
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0 or not os.path.exists(dockq_out):
                failed.append((pdb_id, seed))
                continue

            with open(dockq_out) as f:
                dq = json.load(f)

            results.append({
                'pdb': pdb_id,
                'seed': seed,
                'raw': json.dumps(dq)
            })
            done.add((pdb_id, seed))

            # Clean up
            for f in [cif_dst, pdb_dst, dockq_out]:
                if os.path.exists(f):
                    os.remove(f)

        except Exception as e:
            failed.append((pdb_id, seed))

    # Checkpoint every 10 complexes
    if (idx + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
        print(f"[{idx+1}/{len(df)}] Saved: {len(results)}, Failed: {len(failed)}")

pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
print(f"\nDone. Results: {len(results)}, Failed: {len(failed)}")
