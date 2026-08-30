"""
Script 16: Chai-1 DockQ computation
Input: benchmark_master.tsv, Chai-1 CIF files on Drive
Output: chai1_dockq_results.tsv
70 complexes × 5 trunks = 350 predictions
"""

import subprocess
import json
import os
import pandas as pd
import gemmi

DRIVE_CHAI = 'gdrive:antibody_benchmark/chai1_outputs'
TMP = '/tmp/chai1_dockq'
OUTFILE = 'chai1_dockq_results.tsv'
os.makedirs(TMP, exist_ok=True)

df_meta = pd.read_csv('benchmark_master.tsv', sep='\t')

if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(zip(df_existing['pdb'], df_existing['trunk']))
    results = df_existing.to_dict('records')
    print(f"Resuming: {len(results)} done")
else:
    done = set()
    results = []

failed = []

ls_result = subprocess.run([
    'rclone', 'lsf', DRIVE_CHAI
], capture_output=True, text=True, timeout=60)

pdb_ids = [d.strip('/') for d in ls_result.stdout.strip().split('\n') if d.strip('/')]
print(f"Chai-1 complexes on Drive: {len(pdb_ids)}")

for pdb_id in pdb_ids:
    native_path = f'pdb_structures/{pdb_id}.pdb'
    if not os.path.exists(native_path):
        continue

    for trunk_idx in range(5):
        if (pdb_id, trunk_idx) in done:
            continue

        cif_src = f'{DRIVE_CHAI}/{pdb_id}/trunk_{trunk_idx}/pred.model_idx_0.cif'
        cif_dst = f'{TMP}/{pdb_id}_trunk{trunk_idx}.cif'
        pdb_dst = f'{TMP}/{pdb_id}_trunk{trunk_idx}.pdb'

        try:
            if not os.path.exists(cif_dst):
                r = subprocess.run([
                    'rclone', 'copy', cif_src, TMP
                ], capture_output=True, timeout=90)
                src = f'{TMP}/pred.model_idx_0.cif'
                if os.path.exists(src):
                    os.rename(src, cif_dst)

            if not os.path.exists(cif_dst):
                failed.append((pdb_id, trunk_idx))
                continue

            doc = gemmi.cif.read(cif_dst)
            st = gemmi.make_structure_from_block(doc.sole_block())
            st.write_pdb(pdb_dst)

            dockq_out = f'{TMP}/dq_{pdb_id}_trunk{trunk_idx}.json'
            result = subprocess.run([
                'DockQ', pdb_dst, native_path, '--json', dockq_out
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0 or not os.path.exists(dockq_out):
                failed.append((pdb_id, trunk_idx))
                continue

            with open(dockq_out) as f:
                dq = json.load(f)

            results.append({
                'pdb': pdb_id,
                'trunk': trunk_idx,
                'raw': json.dumps(dq)
            })
            done.add((pdb_id, trunk_idx))

        except Exception as e:
            failed.append((pdb_id, trunk_idx))
            print(f"Failed {pdb_id} trunk{trunk_idx}: {e}")

    pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
    print(f"[{pdb_id}] Done. Total: {len(results)}")

pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
print(f"\nComplete. Results: {len(results)}, Failed: {len(failed)}")
