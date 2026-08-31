"""
Script 19: OF3 Contact position PAE analysis
Same as Script 10 but for OpenFold3.
Uses existing contact_flags from boltz2_contact_positions.tsv
(contact positions are from experimental PDB — same for all models)
PAE extracted from OF3 confidences.json (key: 'pae')
"""

import numpy as np
import pandas as pd
import subprocess
import json
import os

DRIVE_OF3 = 'gdrive:antibody_benchmark/of3_outputs'
TMP = '/tmp/of3_contact_pae'
OUTFILE = 'of3_contact_pae.tsv'
os.makedirs(TMP, exist_ok=True)

df = pd.read_csv('benchmark_master.tsv', sep='\t')
df_contacts = pd.read_csv('boltz2_contact_positions.tsv', sep='\t')
df = df.merge(df_contacts[['pdb','contact_flags','n_contact','contact_fraction']], on='pdb')

# Load checkpoint
if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(df_existing['pdb'])
    results = df_existing.to_dict('records')
    print(f"Resuming: {len(results)} done")
else:
    done = set()
    results = []

failed = []
seed_idx = 0  # Use first seed only

print(f"OF3 Contact PAE for {len(df)} complexes...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    if pdb_id in done:
        continue

    hseq = row['hseq']
    cdr_h3_seq = row['cdr_h3']
    contact_flags = [x == 'True' for x in row['contact_flags'].split(',')]

    if not isinstance(cdr_h3_seq, str) or not isinstance(hseq, str):
        continue

    cdr_h3_idx = hseq.find(cdr_h3_seq)
    if cdr_h3_idx == -1:
        continue

    # Get first seed dir
    ls_result = subprocess.run([
        'rclone', 'lsf', f'{DRIVE_OF3}/{pdb_id}/'
    ], capture_output=True, text=True, timeout=120)

    if ls_result.returncode != 0 or not ls_result.stdout.strip():
        failed.append(pdb_id)
        continue

    seed_dirs = [d.strip('/') for d in ls_result.stdout.strip().split('\n')
                 if d.strip('/').startswith('seed_')]
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
        lseq = row['lseq'] if pd.notna(row['lseq']) else ''
        light_len = len(lseq)
        antigen_start = heavy_len + light_len

        cdr_h3_start = cdr_h3_idx
        contact_indices = [cdr_h3_start + j for j, f in enumerate(contact_flags) if f]
        noncontact_indices = [cdr_h3_start + j for j, f in enumerate(contact_flags) if not f]

        if not contact_indices:
            continue

        pae_contact_ag = pae[contact_indices, :][:, antigen_start:].mean()
        pae_noncontact_ag = pae[noncontact_indices, :][:, antigen_start:].mean() if noncontact_indices else np.nan
        pae_cdr_ag = pae[cdr_h3_start:cdr_h3_start+len(cdr_h3_seq), antigen_start:].mean()

        results.append({
            'pdb': pdb_id,
            'n_contact': sum(contact_flags),
            'n_noncontact': len(contact_flags) - sum(contact_flags),
            'contact_fraction': row['contact_fraction'],
            'pae_contact_ag': pae_contact_ag,
            'pae_noncontact_ag': pae_noncontact_ag,
            'pae_cdr_ag': pae_cdr_ag,
            'ratio_contact_noncontact': pae_contact_ag / pae_noncontact_ag
                if not np.isnan(pae_noncontact_ag) and pae_noncontact_ag > 0 else np.nan
        })
        done.add(pdb_id)

        if os.path.exists(conf_dst):
            os.remove(conf_dst)

        if (i+1) % 50 == 0:
            pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
            print(f"[{i+1}/{len(df)}] Saved: {len(results)}, Failed: {len(failed)}")

    except Exception as e:
        failed.append(pdb_id)

pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
print(f"\nDone: {len(results)} complexes, Failed: {len(failed)}")
print(f"\n=== OF3 CONTACT vs NON-CONTACT PAE ===")
df_result = pd.DataFrame(results)
print(f"Mean PAE CONTACT:     {df_result['pae_contact_ag'].mean():.3f}")
print(f"Mean PAE NON-CONTACT: {df_result['pae_noncontact_ag'].mean():.3f}")
print(f"Ratio contact/non-contact: {df_result['ratio_contact_noncontact'].mean():.3f}")
print(f"Contact PAE < non-contact: {(df_result['ratio_contact_noncontact'] < 1).sum()} / {len(df_result)}")
