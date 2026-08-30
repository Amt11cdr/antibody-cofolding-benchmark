"""
Script 14: OpenFold3 DockQ Computation
With checkpointing — saves every 10 complexes.
Resumes from checkpoint on restart.
"""

import subprocess
import json
import os
import pandas as pd
from datetime import datetime

DRIVE_OF3 = 'gdrive:antibody_benchmark/of3_outputs'
TMP = '/tmp/of3_dockq'
OUTFILE = os.path.expanduser('~/antibody_benchmark/data/of3_dockq_results.tsv')
os.makedirs(TMP, exist_ok=True)

df_meta = pd.read_csv('benchmark_master.tsv', sep='\t')

# Load checkpoint
if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(zip(df_existing['pdb'], df_existing['seed_dir']))
    results = df_existing.to_dict('records')
    print(f"Resuming from checkpoint: {len(results)} results saved")
else:
    done = set()
    results = []

def save_checkpoint():
    pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
    print(f"  Checkpoint saved: {len(results)} results")

print(f"Processing {len(df_meta)} complexes...")

for complex_idx, (_, row) in enumerate(df_meta.iterrows()):
    pdb_id = row['pdb']
    native_path = f'pdb_structures/{pdb_id}.pdb'

    if not os.path.exists(native_path):
        continue

    # List seed folders
    ls_result = subprocess.run([
        'rclone', 'lsf', '--timeout', '2m', f'{DRIVE_OF3}/{pdb_id}/'
    ], capture_output=True, text=True, timeout=150)

    if ls_result.returncode != 0 or not ls_result.stdout.strip():
        continue

    seed_dirs = [d.strip('/') for d in ls_result.stdout.strip().split('\n')
                 if d.strip('/').startswith('seed_')]

    for seed_dir in seed_dirs:
        if (pdb_id, seed_dir) in done:
            continue

        cif_name = f'{pdb_id}_{seed_dir}_sample_1_model.cif'
        cif_dst = f'{TMP}/{pdb_id}_{seed_dir}.cif'

        try:
            if not os.path.exists(cif_dst):
                r = subprocess.run([
                    'rclone', 'copy',
                    f'{DRIVE_OF3}/{pdb_id}/{seed_dir}/{cif_name}',
                    TMP
                ], capture_output=True, timeout=120)
                src = f'{TMP}/{cif_name}'
                if os.path.exists(src):
                    os.rename(src, cif_dst)

            if not os.path.exists(cif_dst):
                continue

            # Convert CIF to PDB for DockQ compatibility
            pdb_dst = cif_dst.replace('.cif', '.pdb')
            try:
                import gemmi
                doc = gemmi.cif.read(cif_dst)
                st = gemmi.make_structure_from_block(doc.sole_block())
                st.write_pdb(pdb_dst)
            except Exception as e:
                continue

            dockq_out = f'{TMP}/dq_{pdb_id}_{seed_dir}.json'
            result = subprocess.run([
                'DockQ', pdb_dst, native_path,
                '--json', dockq_out
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                continue

            with open(dockq_out) as f:
                dq = json.load(f)

            results.append({
                'pdb': pdb_id,
                'seed_dir': seed_dir,
                'raw': json.dumps(dq)
            })
            done.add((pdb_id, seed_dir))

            # Clean up CIF to save disk
            if os.path.exists(cif_dst):
                os.remove(cif_dst)

        except Exception as e:
            print(f"Failed {pdb_id} {seed_dir}: {e}")

    # Checkpoint every 10 complexes
    if (complex_idx + 1) % 10 == 0:
        save_checkpoint()
        print(f"[{complex_idx+1}/{len(df_meta)}] {pdb_id} done")

# Final save
save_checkpoint()
print(f"\nComplete. Total results: {len(results)}")
