"""
Script 22: Build PAE-GAT dataset from OpenFold3 predictions
Same as Script 12 but for OF3 output format.
PAE key: 'pae' from confidences.json (17MB file)
Seed dirs named by actual seed numbers not 1-5.
Output: of3_paegat_dataset/ directory with one .pt file per prediction
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import subprocess
import os
import json

DRIVE_OF3 = 'gdrive:antibody_benchmark/of3_outputs'
TMP = '/tmp/pae_of3_paegat'
OUT_DIR = os.path.expanduser('~/antibody_benchmark/of3_paegat_dataset')
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_TO_IDX['X'] = 20

def aa_onehot(seq):
    one_hot = np.zeros((len(seq), 21), dtype=np.float32)
    for i, aa in enumerate(seq):
        one_hot[i, AA_TO_IDX.get(aa, 20)] = 1.0
    return one_hot

def build_node_features(seq, chain_type):
    n = len(seq)
    onehot = aa_onehot(seq)
    pos = np.arange(n, dtype=np.float32).reshape(-1, 1) / max(n, 1)
    chain_flag = np.full((n, 1), chain_type, dtype=np.float32)
    return np.concatenate([onehot, pos, chain_flag], axis=1)

df_analysis = pd.read_csv('of3_analysis_ready.tsv', sep='\t')
df_meta = pd.read_csv('benchmark_master.tsv', sep='\t')
df = df_analysis.merge(df_meta[['pdb','hseq','lseq','cdr_h3']], on='pdb', how='inner')
df = df[df['DockQ_HA'].notna()].reset_index(drop=True)
print(f"Valid OF3 predictions: {len(df)}")

# Load checkpoint
built_files = set(os.listdir(OUT_DIR))
skipped = 0
built = 0

for i, row in df.iterrows():
    pdb_id = row['pdb']
    seed_dir = row['seed_dir']
    dockq_ha = row['DockQ_HA']
    hseq = row['hseq']
    lseq = row['lseq'] if pd.notna(row['lseq']) else ''
    cdr_h3_seq = row['cdr_h3']
    iptm = row['iptm']

    out_name = f'{pdb_id}_{seed_dir}.pt'
    if out_name in built_files:
        built += 1
        continue

    if not isinstance(hseq, str):
        skipped += 1
        continue

    # Download OF3 PAE JSON
    conf_name = f'{pdb_id}_{seed_dir}_sample_1_confidences.json'
    conf_dst = f'{TMP}/{pdb_id}_{seed_dir}_conf.json'

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
            skipped += 1
            continue

        with open(conf_dst) as f:
            conf_data = json.load(f)

        pae = np.array(conf_data['pae'])

        heavy_len = len(hseq)
        light_len = len(lseq)
        antigen_start = heavy_len + light_len
        antigen_end = pae.shape[0]
        antigen_len = antigen_end - antigen_start

        if antigen_len <= 0:
            skipped += 1
            os.remove(conf_dst)
            continue

        x_heavy = build_node_features(hseq, 0)
        x_light = build_node_features(lseq, 1) if light_len > 0 else np.zeros((0, 23), dtype=np.float32)
        x_antigen = build_node_features('X' * antigen_len, 2)
        x = np.concatenate([x_heavy, x_light, x_antigen], axis=0)

        antigen_offset = heavy_len + light_len
        src_nodes, dst_nodes, edge_weights = [], [], []

        # Heavy <-> Antigen
        for hi in range(heavy_len):
            for aj in range(antigen_len):
                src_nodes.append(hi)
                dst_nodes.append(antigen_offset + aj)
                edge_weights.append(pae[hi, antigen_start + aj])
                src_nodes.append(antigen_offset + aj)
                dst_nodes.append(hi)
                edge_weights.append(pae[antigen_start + aj, hi])

        # Light <-> Antigen
        if light_len > 0:
            for li in range(light_len):
                for aj in range(antigen_len):
                    src_nodes.append(heavy_len + li)
                    dst_nodes.append(antigen_offset + aj)
                    edge_weights.append(pae[heavy_len + li, antigen_start + aj])
                    src_nodes.append(antigen_offset + aj)
                    dst_nodes.append(heavy_len + li)
                    edge_weights.append(pae[antigen_start + aj, heavy_len + li])

            # Heavy <-> Light
            for hi in range(heavy_len):
                for li in range(light_len):
                    src_nodes.append(hi)
                    dst_nodes.append(heavy_len + li)
                    edge_weights.append(pae[hi, heavy_len + li])
                    src_nodes.append(heavy_len + li)
                    dst_nodes.append(hi)
                    edge_weights.append(pae[heavy_len + li, hi])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([dockq_ha], dtype=torch.float32)

        cdr_h3_idx = hseq.find(cdr_h3_seq) if isinstance(cdr_h3_seq, str) else -1
        cdr_h3_mask = torch.zeros(len(x), dtype=torch.bool)
        if cdr_h3_idx >= 0:
            cdr_h3_mask[cdr_h3_idx:cdr_h3_idx + len(cdr_h3_seq)] = True

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            iptm=torch.tensor([iptm], dtype=torch.float32),
            cdr_h3_mask=cdr_h3_mask,
            heavy_len=heavy_len,
            light_len=light_len,
            antigen_len=antigen_len,
            pdb=pdb_id,
            seed_dir=seed_dir
        )

        torch.save(data, f'{OUT_DIR}/{out_name}')
        built_files.add(out_name)
        built += 1

        # Delete local PAE JSON to save disk
        if os.path.exists(conf_dst):
            os.remove(conf_dst)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(df)}] Built: {built}, Skipped: {skipped}")

    except Exception as e:
        skipped += 1
        if os.path.exists(conf_dst):
            os.remove(conf_dst)
        if skipped < 5:
            print(f"Error {pdb_id} {seed_dir}: {e}")

print(f"\nDone. Built: {built}, Skipped: {skipped}")
print(f"Dataset: {OUT_DIR}")
