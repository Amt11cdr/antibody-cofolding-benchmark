"""
Script 10: Contact position PAE analysis
Input: benchmark_master.tsv, boltz2_contact_positions.tsv,
       Google Drive (PAE matrices via rclone)
Output: boltz2_contact_pae.tsv

For each complex, extracts PAE values specifically at:
- Contact CDR-H3 positions (residues touching antigen)
- Non-contact CDR-H3 positions (residues not touching antigen)

Key finding: Contact PAE (13.051) barely differs from
non-contact PAE (12.484) — ratio 1.06.
Despite contact positions being 3-4x harder to predict correctly,
the model assigns almost identical uncertainty to contact and
non-contact CDR-H3 residues.

This proves Angle 2 (contact position density) of the mechanistic
framework: the model cannot distinguish which CDR-H3 residues
matter most for binding in its uncertainty representation.

Requirements: pip install numpy pandas && brew install rclone
"""

import numpy as np
import pandas as pd
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

DRIVE_OUT = 'gdrive:antibody_benchmark/outputs'
TMP = '/tmp/pae_contact'
os.makedirs(TMP, exist_ok=True)

df = pd.read_csv('benchmark_master.tsv', sep='\t')
df_contacts = pd.read_csv('boltz2_contact_positions.tsv', sep='\t')
df = df.merge(df_contacts[['pdb', 'contact_flags', 'n_contact', 'contact_fraction']], on='pdb')

results = []
seed = 1

print(f"Extracting contact vs non-contact PAE for {len(df)} complexes...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    hseq = row['hseq']
    cdr_h3_seq = row['cdr_h3']
    contact_flags = [x == 'True' for x in row['contact_flags'].split(',')]

    if not isinstance(cdr_h3_seq, str) or not isinstance(hseq, str):
        continue

    cdr_h3_idx = hseq.find(cdr_h3_seq)
    if cdr_h3_idx == -1:
        continue

    pae_dst = f'{TMP}/{pdb_id}_pae.npz'

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

        if not os.path.exists(pae_dst):
            continue

        pae = np.load(pae_dst)['pae']

        # Chain boundaries
        heavy_len = len(hseq)
        lseq = row['lseq'] if pd.notna(row['lseq']) else ''
        light_len = len(lseq)
        antigen_start = heavy_len + light_len

        # CDR-H3 indices
        cdr_h3_start = cdr_h3_idx
        cdr_h3_end = cdr_h3_idx + len(cdr_h3_seq)

        # Split into contact and non-contact
        contact_indices = [cdr_h3_start + j for j, f in enumerate(contact_flags) if f]
        noncontact_indices = [cdr_h3_start + j for j, f in enumerate(contact_flags) if not f]

        if not contact_indices:
            continue

        pae_contact_ag = pae[contact_indices, :][:, antigen_start:].mean()
        pae_noncontact_ag = pae[noncontact_indices, :][:, antigen_start:].mean() \
                            if noncontact_indices else np.nan
        pae_cdr_ag = pae[cdr_h3_start:cdr_h3_end, antigen_start:].mean()

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

        if (i+1) % 50 == 0:
            print(f"  {i+1}/304 done...")

    except Exception as e:
        print(f"Failed {pdb_id}: {e}")

df_result = pd.DataFrame(results)
df_result.to_csv('boltz2_contact_pae.tsv', sep='\t', index=False)

print(f"\nDone: {len(df_result)} complexes")
print(f"\n=== CONTACT vs NON-CONTACT PAE ===")
print(f"Mean PAE CONTACT positions/antigen:     {df_result['pae_contact_ag'].mean():.3f}")
print(f"Mean PAE NON-CONTACT positions/antigen: {df_result['pae_noncontact_ag'].mean():.3f}")
print(f"Mean PAE all CDR-H3/antigen:            {df_result['pae_cdr_ag'].mean():.3f}")
print(f"Ratio contact/non-contact:              {df_result['ratio_contact_noncontact'].mean():.3f}")
print(f"Contact PAE < non-contact PAE:          {(df_result['ratio_contact_noncontact'] < 1).sum()} / {len(df_result)}")
