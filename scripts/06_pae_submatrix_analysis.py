"""
Script 6: PAE submatrix analysis — residue imbalance quantification
Input: benchmark_master.tsv, Google Drive (PAE matrices via rclone)
Output: boltz2_pae_analysis.tsv

For each complex (seed 1), extracts:
- Mean PAE at CDR-H3/antigen submatrix
- Mean PAE at framework/antigen submatrix
- Ratio CDR-H3/framework PAE
- Residue imbalance ratio (framework count / CDR-H3 count)

Key finding: CDR-H3 PAE < framework PAE in 74.7% of complexes
despite worse CDR-H3 structural accuracy.
This proves Angle 3 (residue imbalance) of the mechanistic framework.
"""

import numpy as np
import pandas as pd
import json
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

DRIVE_OUT = 'gdrive:antibody_benchmark/outputs'
TMP = '/tmp/pae_batch'
os.makedirs(TMP, exist_ok=True)

df = pd.read_csv('benchmark_master.tsv', sep='\t')
results = []
failed = []

print(f"Processing {len(df)} complexes (seed 1 only)...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    hseq = row['hseq']
    cdr_h3_seq = row['cdr_h3']
    seed = 1

    if not isinstance(cdr_h3_seq, str) or not isinstance(hseq, str):
        failed.append(pdb_id)
        continue

    cdr_h3_idx = hseq.find(cdr_h3_seq)
    if cdr_h3_idx == -1:
        failed.append(pdb_id)
        continue

    pae_dst = f'{TMP}/{pdb_id}_pae.npz'
    conf_dst = f'{TMP}/{pdb_id}_conf.json'

    try:
        if not os.path.exists(pae_dst):
            subprocess.run([
                'rclone', 'copy',
                f'{DRIVE_OUT}/{pdb_id}/seed{seed}/pae_{pdb_id}_model_0.npz',
                TMP
            ], capture_output=True, timeout=60)
            src = f'{TMP}/pae_{pdb_id}_model_0.npz'
            if os.path.exists(src):
                os.rename(src, pae_dst)

        if not os.path.exists(conf_dst):
            subprocess.run([
                'rclone', 'copy',
                f'{DRIVE_OUT}/{pdb_id}/seed{seed}/confidence_{pdb_id}_model_0.json',
                TMP
            ], capture_output=True, timeout=60)
            src = f'{TMP}/confidence_{pdb_id}_model_0.json'
            if os.path.exists(src):
                os.rename(src, conf_dst)

        if not os.path.exists(pae_dst) or not os.path.exists(conf_dst):
            failed.append(pdb_id)
            continue

        pae = np.load(pae_dst)['pae']
        with open(conf_dst) as f:
            conf = json.load(f)

        # Chain boundaries
        heavy_len = len(hseq)
        lseq = row['lseq'] if pd.notna(row['lseq']) else ''
        light_len = len(lseq)
        antigen_start = heavy_len + light_len

        # CDR-H3 indices
        cdr_h3_start = cdr_h3_idx
        cdr_h3_end = cdr_h3_idx + len(cdr_h3_seq)

        # PAE submatrices
        pae_cdr_ag = pae[cdr_h3_start:cdr_h3_end, antigen_start:].mean()
        fw_idx = list(range(0, cdr_h3_start)) + list(range(cdr_h3_end, heavy_len))
        pae_fw_ag = pae[fw_idx, :][:, antigen_start:].mean()
        pae_heavy_ag = pae[0:heavy_len, antigen_start:].mean()

        results.append({
            'pdb': pdb_id,
            'cdr_h3_len': len(cdr_h3_seq),
            'has_light': pd.notna(row['lseq']),
            'iptm': conf['iptm'],
            'confidence_score': conf['confidence_score'],
            'pae_cdr_ag': pae_cdr_ag,
            'pae_fw_ag': pae_fw_ag,
            'pae_heavy_ag': pae_heavy_ag,
            'ratio_cdr_fw': pae_cdr_ag / pae_fw_ag if pae_fw_ag > 0 else None,
            'n_framework': len(fw_idx),
            'n_cdr_h3': len(cdr_h3_seq),
            'imbalance_ratio': len(fw_idx) / len(cdr_h3_seq)
        })

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/304] done")

    except Exception as e:
        failed.append(pdb_id)

df_pae = pd.DataFrame(results)
df_pae.to_csv('boltz2_pae_analysis.tsv', sep='\t', index=False)

print(f"\nDone. Computed: {len(df_pae)}, Failed: {len(failed)}")
print(f"\nKey results:")
print(f"Mean PAE CDR-H3/Ag: {df_pae['pae_cdr_ag'].mean():.3f}")
print(f"Mean PAE FW/Ag:     {df_pae['pae_fw_ag'].mean():.3f}")
print(f"Mean ratio CDR-H3/FW: {df_pae['ratio_cdr_fw'].mean():.3f}")
print(f"CDR-H3 PAE < FW PAE: {(df_pae['ratio_cdr_fw'] < 1).sum()} / {len(df_pae)} ({100*(df_pae['ratio_cdr_fw'] < 1).mean():.1f}%)")
