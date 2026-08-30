"""
Script 12: Build PAE-GAT dataset from Boltz-2 predictions
Full inter-chain PAE graph as input — not just CDR-H3/antigen.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import subprocess
import os

DRIVE_OUT = 'gdrive:antibody_benchmark/outputs'
TMP = '/tmp/pae_paegat'
OUT_DIR = os.path.expanduser('~/antibody_benchmark/paegat_dataset')
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_TO_IDX['X'] = 20

def aa_onehot(seq):
    one_hot = np.zeros((len(seq), 21), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, 20)
        one_hot[i, idx] = 1.0
    return one_hot

def build_node_features(seq, chain_type):
    n = len(seq)
    onehot = aa_onehot(seq)
    pos = np.arange(n, dtype=np.float32).reshape(-1, 1) / max(n, 1)
    chain_flag = np.full((n, 1), chain_type, dtype=np.float32)
    return np.concatenate([onehot, pos, chain_flag], axis=1)

df_analysis = pd.read_csv('boltz2_analysis_ready.tsv', sep='\t')
df_meta = pd.read_csv('benchmark_master.tsv', sep='\t')
df = df_analysis.merge(df_meta[['pdb','hseq','lseq','cdr_h3','antigen_chain']], on='pdb', how='inner')
df = df[df['DockQ_HA'].notna()].reset_index(drop=True)
print(f"Valid predictions: {len(df)}")

skipped = 0
built = 0

for i, row in df.iterrows():
    pdb_id = row['pdb']
    seed = row['seed']
    dockq_ha = row['DockQ_HA']
    hseq = row['hseq']
    lseq = row['lseq'] if pd.notna(row['lseq']) else ''
    cdr_h3_seq = row['cdr_h3']
    iptm = row['iptm']

    out_path = f'{OUT_DIR}/{pdb_id}_seed{seed}.pt'
    if os.path.exists(out_path):
        built += 1
        continue

    if not isinstance(hseq, str):
        skipped += 1
        continue

    pae_dst = f'{TMP}/{pdb_id}_seed{seed}_pae.npz'
    if not os.path.exists(pae_dst):
        r = subprocess.run([
            'rclone', 'copy',
            f'{DRIVE_OUT}/{pdb_id}/seed{seed}/pae_{pdb_id}_model_0.npz',
            TMP
        ], capture_output=True, timeout=180)
        src = f'{TMP}/pae_{pdb_id}_model_0.npz'
        if os.path.exists(src):
            os.rename(src, pae_dst)

    if not os.path.exists(pae_dst):
        skipped += 1
        continue

    try:
        pae = np.load(pae_dst)['pae']

        heavy_len = len(hseq)
        light_len = len(lseq)
        antigen_start = heavy_len + light_len
        antigen_end = pae.shape[0]
        antigen_len = antigen_end - antigen_start

        if antigen_len <= 0:
            skipped += 1
            continue

        x_heavy = build_node_features(hseq, 0)
        x_light = build_node_features(lseq, 1) if light_len > 0 else np.zeros((0, 23), dtype=np.float32)
        x_antigen = build_node_features('X' * antigen_len, 2)
        x = np.concatenate([x_heavy, x_light, x_antigen], axis=0)

        heavy_offset = 0
        light_offset = heavy_len
        antigen_offset = heavy_len + light_len

        src_nodes, dst_nodes, edge_weights = [], [], []

        for hi in range(heavy_len):
            for aj in range(antigen_len):
                src_nodes.append(heavy_offset + hi)
                dst_nodes.append(antigen_offset + aj)
                edge_weights.append(pae[hi, antigen_start + aj])
                src_nodes.append(antigen_offset + aj)
                dst_nodes.append(heavy_offset + hi)
                edge_weights.append(pae[antigen_start + aj, hi])

        if light_len > 0:
            for li in range(light_len):
                for aj in range(antigen_len):
                    src_nodes.append(light_offset + li)
                    dst_nodes.append(antigen_offset + aj)
                    edge_weights.append(pae[heavy_len + li, antigen_start + aj])
                    src_nodes.append(antigen_offset + aj)
                    dst_nodes.append(light_offset + li)
                    edge_weights.append(pae[antigen_start + aj, heavy_len + li])

            for hi in range(heavy_len):
                for li in range(light_len):
                    src_nodes.append(heavy_offset + hi)
                    dst_nodes.append(light_offset + li)
                    edge_weights.append(pae[hi, heavy_len + li])
                    src_nodes.append(light_offset + li)
                    dst_nodes.append(heavy_offset + hi)
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
            seed=int(seed)
        )

        torch.save(data, out_path)
        built += 1

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(df)}] Built: {built}, Skipped: {skipped}")

    except Exception as e:
        skipped += 1
        if skipped < 5:
            print(f"Error {pdb_id} seed {seed}: {e}")

print(f"\nDone. Built: {built}, Skipped: {skipped}")
