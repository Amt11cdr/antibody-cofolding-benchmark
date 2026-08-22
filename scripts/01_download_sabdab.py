"""
Script 1: Download and filter SAbDab database
Produces: sabdab_summary.tsv, sabdab_post_cutoff.tsv, sabdab_post_cutoff_dedup.tsv

Filters applied:
- X-ray crystallography only
- Protein antigens only
- Resolution <= 2.5 Angstrom
- Heavy chain present
- Antigen chain present
- Time split: post Boltz-2 training cutoff (June 1, 2023)
"""

import pandas as pd
import urllib.request

# Download SAbDab summary
print("Downloading SAbDab summary...")
urllib.request.urlretrieve(
    "https://huggingface.co/datasets/RosettaCommons/SAbDab/resolve/main/sabdab_summary_all.tsv",
    "sabdab_summary.tsv"
)

df = pd.read_csv('sabdab_summary.tsv', sep='\t', low_memory=False)
print(f"Raw SAbDab entries: {len(df)}")

# Apply filters
xray = df[df['method'] == 'X-RAY DIFFRACTION'].copy()
print(f"After X-ray filter: {len(xray)}")

protein = xray[xray['antigen_type'] == 'protein'].copy()
print(f"After protein antigen filter: {len(protein)}")

protein['resolution'] = pd.to_numeric(protein['resolution'], errors='coerce')
highres = protein[protein['resolution'] <= 2.5].copy()
print(f"After resolution <= 2.5A filter: {len(highres)}")

has_heavy = highres[highres['Hchain'].notna()].copy()
filtered = has_heavy[has_heavy['antigen_chain'].notna()].copy()
print(f"After chain filters: {len(filtered)}")

# Time split at Boltz-2 training cutoff
filtered['date_parsed'] = pd.to_datetime(filtered['date'], format='%m/%d/%y', errors='coerce')
BOLTZ2_CUTOFF = '2023-06-01'

pre = filtered[filtered['date_parsed'] < BOLTZ2_CUTOFF]
post = filtered[filtered['date_parsed'] >= BOLTZ2_CUTOFF]

print(f"\nBoltz-2 training cutoff: {BOLTZ2_CUTOFF}")
print(f"Pre-cutoff (training seen): {len(pre)}")
print(f"Post-cutoff (held-out): {len(post)}")

pre.to_csv('sabdab_pre_cutoff.tsv', sep='\t', index=False)
post.to_csv('sabdab_post_cutoff.tsv', sep='\t', index=False)

# Deduplicate at PDB level
post_dedup = post.drop_duplicates(subset='pdb', keep='first').copy()
post_dedup['has_light'] = post_dedup['Lchain'].notna()
post_dedup.to_csv('sabdab_post_cutoff_dedup.tsv', sep='\t', index=False)

print(f"\nFinal benchmark set: {len(post_dedup)} unique PDB structures")
print(f"Full antibodies (VH+VL): {post_dedup['has_light'].sum()}")
print(f"Heavy-only (VHH): {(~post_dedup['has_light']).sum()}")
