"""
Script 17: OpenFold3 Region-wide PAE analysis — all six CDR loops
Input: benchmark_master.tsv, OF3 confidences.json on Drive (seed 1)
Output: of3_region_pae.tsv

Extracts mean PAE at antigen interface for all six CDR loops.
OF3 PAE key: 'pae' (not 'token_pair_pae' which is Protenix)
Local copies deleted after extraction — originals on Drive.
"""

import numpy as np
import pandas as pd
import subprocess
import json
import os

DRIVE_OF3 = 'gdrive:antibody_benchmark/of3_outputs'
TMP = '/tmp/pae_region_of3'
OUTFILE = 'of3_region_pae.tsv'
os.makedirs(TMP, exist_ok=True)

df = pd.read_csv('benchmark_master.tsv', sep='\t')

if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(df_existing['pdb'])
    results = df_existing.to_dict('records')
    print(f"Resuming: {len(results)} done")
else:
    done = set()
    results = []

failed = []

def get_cdr_pae(seq, cdr_seq, chain_offset, antigen_start, antigen_end, pae):
    if not isinstance(cdr_seq, str) or not isinstance(seq, str):
        return np.nan
    idx = seq.find(cdr_seq)
    if idx == -1:
        return np.nan
    cdr_start = chain_offset + idx
    cdr_end = cdr_start + len(cdr_seq)
    return pae[cdr_start:cdr_end, antigen_start:antigen_end].mean()

def get_fw_pae(seq, cdrs, chain_offset, antigen_start, antigen_end, pae):
    n = len(seq)
    mask = np.ones(n, dtype=bool)
    for cdr in cdrs:
        if isinstance(cdr, str):
            idx = seq.find(cdr)
            if idx >= 0:
                mask[idx:idx+len(cdr)] = False
    fw_indices = chain_offset + np.where(mask)[0]
    if len(fw_indices) == 0:
        return np.nan
    return pae[fw_indices, :][:, antigen_start:antigen_end].mean()

print(f"OF3 Region-wide PAE analysis for {len(df)} complexes...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    if pdb_id in done:
        continue

    hseq = row['hseq']
    lseq = row['lseq'] if pd.notna(row['lseq']) else ''

    if not isinstance(hseq, str):
        continue

    ls_result = subprocess.run([
        'rclone', 'lsf', f'{DRIVE_OF3}/{pdb_id}/'
    ], capture_output=True, text=True, timeout=120)

    if ls_result.returncode != 0 or not ls_result.stdout.strip():
        failed.append(pdb_id)
        continue

    seed_dirs = [d.strip('/') for d in ls_result.stdout.strip().split('\n') if d.strip('/').startswith('seed_')]
    if not seed_dirs:
        failed.append(pdb_id)
        continue

    seed_dir = seed_dirs[0]
    conf_name = f'{pdb_id}_{seed_dir}_sample_1_confidences.json'
    conf_dst = f'{TMP}/{pdb_id}_conf.json'

    try:
        if not os.path.exists(conf_dst):
            r = subprocess.run([
                'rclone', 'copy',
                f'{DRIVE_OF3}/{pdb_id}/{seed_dir}/{conf_name}',
                TMP
            ], capture_output=True, timeout=180)
            src = f'{TMP}/{conf_name}'
            if os.path.exists(src):
                os.rename(src, conf_dst)

        if not os.path.exists(conf_dst):
            failed.append(pdb_id)
            continue

        with open(conf_dst) as f:
            conf_data = json.load(f)

        pae = np.array(conf_data['pae'])

        heavy_len = len(hseq)
        light_len = len(lseq)
        antigen_start = heavy_len + light_len
        antigen_end = pae.shape[0]

        heavy_cdrs = [row.get('cdr_h1'), row.get('cdr_h2'), row.get('cdr_h3')]

        entry = {
            'pdb': pdb_id,
            'has_light': light_len > 0,
            'pae_h1_ag': get_cdr_pae(hseq, row.get('cdr_h1'), 0, antigen_start, antigen_end, pae),
            'pae_h2_ag': get_cdr_pae(hseq, row.get('cdr_h2'), 0, antigen_start, antigen_end, pae),
            'pae_h3_ag': get_cdr_pae(hseq, row.get('cdr_h3'), 0, antigen_start, antigen_end, pae),
            'pae_fw_ag': get_fw_pae(hseq, heavy_cdrs, 0, antigen_start, antigen_end, pae),
        }

        if light_len > 0:
            light_cdrs = [row.get('cdr_l1'), row.get('cdr_l2'), row.get('cdr_l3')]
            entry.update({
                'pae_l1_ag': get_cdr_pae(lseq, row.get('cdr_l1'), heavy_len, antigen_start, antigen_end, pae),
                'pae_l2_ag': get_cdr_pae(lseq, row.get('cdr_l2'), heavy_len, antigen_start, antigen_end, pae),
                'pae_l3_ag': get_cdr_pae(lseq, row.get('cdr_l3'), heavy_len, antigen_start, antigen_end, pae),
                'pae_lfw_ag': get_fw_pae(lseq, light_cdrs, heavy_len, antigen_start, antigen_end, pae),
            })
        else:
            entry.update({'pae_l1_ag': np.nan, 'pae_l2_ag': np.nan,
                         'pae_l3_ag': np.nan, 'pae_lfw_ag': np.nan})

        results.append(entry)
        done.add(pdb_id)

        # Delete local copy — original on Drive
        if os.path.exists(conf_dst):
            os.remove(conf_dst)

        if (i+1) % 50 == 0:
            pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
            print(f"[{i+1}/{len(df)}] Saved: {len(results)}, Failed: {len(failed)}")

    except Exception as e:
        failed.append(pdb_id)
        if len(failed) < 5:
            print(f"Error {pdb_id}: {e}")

pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
print(f"\nDone. Computed: {len(results)}, Failed: {len(failed)}")
print(f"\nMean PAE by region (OF3):")
df_out = pd.DataFrame(results)
for col in ['pae_h1_ag','pae_h2_ag','pae_h3_ag','pae_fw_ag','pae_l1_ag','pae_l2_ag','pae_l3_ag','pae_lfw_ag']:
    print(f"  {col}: {df_out[col].mean():.3f}")
