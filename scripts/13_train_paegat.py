"""
Script 13: PAE-GAT Version 0 Training
Train on Boltz-2 data, 80/20 complex-level split.
Simple GATv2 on full inter-chain PAE graph.
Output: DockQ_HA regression, compare ECE vs ipTM baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy import stats
import glob

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

DATASET_DIR = os.path.expanduser('~/antibody_benchmark/paegat_dataset')
RESULTS_DIR = os.path.expanduser('~/antibody_benchmark/paegat_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load all graph files
all_files = glob.glob(f'{DATASET_DIR}/*.pt')
print(f"Total graphs: {len(all_files)}")

# Complex-level split to avoid data leakage
# Extract unique complex IDs
pdb_ids = list(set([os.path.basename(f).split('_seed')[0] for f in all_files]))
train_pdbs, test_pdbs = train_test_split(pdb_ids, test_size=0.2, random_state=42)
print(f"Train complexes: {len(train_pdbs)}, Test complexes: {len(test_pdbs)}")

train_files = [f for f in all_files if os.path.basename(f).split('_seed')[0] in set(train_pdbs)]
test_files = [f for f in all_files if os.path.basename(f).split('_seed')[0] in set(test_pdbs)]
print(f"Train graphs: {len(train_files)}, Test graphs: {len(test_files)}")

# Load datasets
def load_graphs(files):
    graphs = []
    for f in files:
        try:
            data = torch.load(f, weights_only=False)
            graphs.append(data)
        except:
            pass
    return graphs

print("Loading train graphs...")
train_data = load_graphs(train_files)
print("Loading test graphs...")
test_data = load_graphs(test_files)
print(f"Loaded: {len(train_data)} train, {len(test_data)} test")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# PAE-GAT Model
class PAEGAT(nn.Module):
    def __init__(self, in_channels=23, hidden_channels=64, heads=4, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Layer 1: each node attends to neighbours, PAE as edge bias
        self.conv1 = GATv2Conv(
            hidden_channels, 
            hidden_channels // heads, 
            heads=heads,
            edge_dim=1,
            dropout=dropout
        )
        
        # Layer 2
        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout
        )
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # DockQ_HA is 0-1
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Message passing
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output
        out = self.output(x)
        return out.squeeze(-1)

model = PAEGAT().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

# Training
def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    preds, labels, iptms = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
            iptms.extend(batch.iptm.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(iptms)

def compute_ece(confidence, accuracy, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() > 0:
            bin_conf = confidence[mask].mean()
            bin_acc = accuracy[mask].mean()
            ece += mask.mean() * abs(bin_conf - bin_acc)
    return ece

print("\nTraining PAE-GAT...")
best_loss = float('inf')

for epoch in range(20):
    train_loss = train_epoch(train_loader)
    
    if (epoch + 1) % 5 == 0:
        preds, labels, iptms = evaluate(test_loader)
        
        # Spearman correlation
        r_paegat, _ = stats.spearmanr(preds, labels)
        r_iptm, _ = stats.spearmanr(iptms, labels)
        
        # ECE
        acceptable = (labels >= 0.23).astype(float)
        ece_paegat = compute_ece(preds, acceptable)
        ece_iptm = compute_ece(iptms, acceptable)
        
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
              f"ECE PAE-GAT: {ece_paegat:.4f} vs ipTM: {ece_iptm:.4f} | "
              f"r PAE-GAT: {r_paegat:.3f} vs ipTM: {r_iptm:.3f}")
        
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f'{RESULTS_DIR}/paegat_best.pt')

# Final evaluation
print("\nFinal evaluation on test set:")
preds, labels, iptms = evaluate(test_loader)

r_paegat, p_paegat = stats.spearmanr(preds, labels)
r_iptm, p_iptm = stats.spearmanr(iptms, labels)

acceptable = (labels >= 0.23).astype(float)
ece_paegat = compute_ece(preds, acceptable)
ece_iptm = compute_ece(iptms, acceptable)

print(f"\nSpearman r — PAE-GAT: {r_paegat:.3f} vs ipTM: {r_iptm:.3f}")
print(f"ECE — PAE-GAT: {ece_paegat:.4f} vs ipTM: {ece_iptm:.4f}")
print(f"ECE improvement: {((ece_iptm - ece_paegat) / ece_iptm * 100):.1f}%")

# Save results
results_df = pd.DataFrame({
    'pdb': [d.pdb for d in test_data],
    'seed': [d.seed for d in test_data],
    'dockq_ha': labels,
    'paegat_pred': preds,
    'iptm': iptms
})
results_df.to_csv(f'{RESULTS_DIR}/paegat_test_results.tsv', sep='\t', index=False)
print(f"\nResults saved to {RESULTS_DIR}")
