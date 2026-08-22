"""
Script 11: Parse DockQ results and merge with confidence scores
Input: boltz2_dockq_results.tsv, benchmark_master.tsv,
       Google Drive (confidence JSONs via rclone)
Output: boltz2_dockq_parsed.tsv, boltz2_analysis_ready.tsv

Parses raw DockQ JSON output to extract:
- DockQ_HA: heavy chain-antigen interface DockQ (primary metric)
- DockQ_LA: light chain-antigen interface DockQ
- iRMSD_HA: interface RMSD at heavy-antigen interface
- fnat_HA: fraction of native contacts at heavy-antigen interface

Then merges with confidence scores (ipTM, pTM, pLDDT) to create
the analysis-ready dataset for calibration analysis.

Key finding after merge:
- Spearman r (ipTM vs DockQ_HA) = 0.632
- ECE = 0.32
- FPR at ipTM >= 0.7 = 44.6%

Note: DockQ computation was run on Google Colab (CPU runtime).
The raw results TSV was downloaded from Drive before running this script.

Requirements: pip install pandas numpy scipy
"""

import json
import os
import pandas as pd
import numpy as np
import subprocess

# ── Step 1: Parse raw DockQ JSON outputs ──────────────────────────────────────
df_raw = pd.read_csv('boltz2_dockq_results.tsv', sep='\t')
df_meta = pd.read_csv('benchmark_master.tsv', sep='\t')

parsed_results = []

for _, row in df_raw.iterrows():
    pdb_id = row['pdb']
    seed = row['seed']

    try:
        raw = json.loads(row['raw'])
        best_result = raw.get('best_result', {})

        # Get chain IDs from metadata
        meta = df_meta[df_meta['pdb'] == pdb_id].iloc[0]
        hchain = meta['Hchain']
        lchain = meta['Lchain'] if pd.notna(meta['Lchain']) else None
        antigen_chains = str(meta['antigen_chain']).split('|')
        antigen_chain = antigen_chains[0].strip()

        # Heavy-antigen interface
        ha_key = f"{hchain}{antigen_chain}"
        ah_key = f"{antigen_chain}{hchain}"
        ha = best_result.get(ha_key) or best_result.get(ah_key) or {}

        # Light-antigen interface
        la = {}
        if lchain:
            la_key = f"{lchain}{antigen_chain}"
            al_key = f"{antigen_chain}{lchain}"
            la = best_result.get(la_key) or best_result.get(al_key) or {}

        # Best single interface DockQ
        all_dockq = [v.get('DockQ', 0) for v in best_result.values()
                     if isinstance(v, dict)]
        best_single = max(all_dockq) if all_dockq else None

        parsed_results.append({
            'pdb': pdb_id,
            'seed': seed,
            'DockQ_HA': ha.get('DockQ'),
            'iRMSD_HA': ha.get('iRMSD'),
            'fnat_HA': ha.get('fnat'),
            'LRMSD_HA': ha.get('LRMSD'),
            'DockQ_LA': la.get('DockQ'),
            'iRMSD_LA': la.get('iRMSD'),
        })

    except Exception as e:
        print(f"Failed {pdb_id} seed {seed}: {e}")

df_parsed = pd.DataFrame(parsed_results)
df_parsed.to_csv('boltz2_dockq_parsed.tsv', sep='\t', index=False)
print(f"Parsed DockQ: {len(df_parsed)} results")
print(f"DockQ_HA valid: {df_parsed['DockQ_HA'].notna().sum()}")

# ── Step 2: Load confidence scores from Drive ──────────────────────────────────
DRIVE_OUT = 'gdrive:antibody_benchmark/outputs'
TMP = '/tmp/conf_scores'
os.makedirs(TMP, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
conf_results = []

print(f"\nLoading confidence scores for {len(df_meta)} complexes...")

for _, row in df_meta.iterrows():
    pdb_id = row['pdb']
    for seed in SEEDS:
        conf_path = f'{TMP}/{pdb_id}_seed{seed}_conf.json'

        try:
            if not os.path.exists(conf_path):
                subprocess.run([
                    'rclone', 'copy',
                    f'{DRIVE_OUT}/{pdb_id}/seed{seed}/confidence_{pdb_id}_model_0.json',
                    TMP
                ], capture_output=True, timeout=60)
                src = f'{TMP}/confidence_{pdb_id}_model_0.json'
                if os.path.exists(src):
                    os.rename(src, conf_path)

            if not os.path.exists(conf_path):
                continue

            with open(conf_path) as f:
                conf = json.load(f)

            conf_results.append({
                'pdb': pdb_id,
                'seed': seed,
                'iptm': conf.get('iptm'),
                'ptm': conf.get('ptm'),
                'confidence_score': conf.get('confidence_score'),
                'complex_plddt': conf.get('complex_plddt'),
                'complex_iplddt': conf.get('complex_iplddt'),
            })

        except Exception as e:
            print(f"Failed {pdb_id} seed {seed}: {e}")

df_conf = pd.DataFrame(conf_results)
print(f"Confidence scores loaded: {len(df_conf)}")

# ── Step 3: Merge DockQ + confidence scores ────────────────────────────────────
df_analysis = df_parsed.merge(df_conf, on=['pdb', 'seed'], how='inner')
df_analysis.to_csv('boltz2_analysis_ready.tsv', sep='\t', index=False)

print(f"\nAnalysis-ready dataset: {len(df_analysis)} rows")
print(f"Columns: {list(df_analysis.columns)}")
print(f"\nKey stats:")
print(f"DockQ_HA mean: {df_analysis['DockQ_HA'].mean():.3f}")
print(f"ipTM mean: {df_analysis['iptm'].mean():.3f}")
