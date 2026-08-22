"""
Script 4: Annotate CDR regions using ANARCI with IMGT numbering
Input: sequences.tsv
Output: cdr_annotations.tsv

Uses ANARCI to assign IMGT numbering to each heavy chain sequence
and extract CDR-H1 (positions 27-38), CDR-H2 (56-65), CDR-H3 (105-117).

Requirements: pip install anarci && brew install hmmer
"""

import pandas as pd
from anarci import anarci

df = pd.read_csv('sequences.tsv', sep='\t')

results = []
failed = []

print(f"Annotating CDRs for {len(df)} complexes...")

for _, row in df.iterrows():
    pdb_id = row['pdb']
    hseq = row['hseq']

    if not isinstance(hseq, str) or len(hseq) < 50:
        failed.append(pdb_id)
        continue

    try:
        numbered, alignment_details, hit_tables = anarci(
            [('heavy', hseq)],
            scheme='imgt',
            output=False
        )

        if numbered[0] is None:
            failed.append(pdb_id)
            continue

        numbering = numbered[0][0][0]

        cdr_h1 = ''.join([aa for pos, aa in numbering
                          if 27 <= pos[0] <= 38 and aa != '-'])
        cdr_h2 = ''.join([aa for pos, aa in numbering
                          if 56 <= pos[0] <= 65 and aa != '-'])
        cdr_h3 = ''.join([aa for pos, aa in numbering
                          if 105 <= pos[0] <= 117 and aa != '-'])

        results.append({
            'pdb': pdb_id,
            'cdr_h1': cdr_h1,
            'cdr_h2': cdr_h2,
            'cdr_h3': cdr_h3,
            'cdr_h3_len': len(cdr_h3),
        })

    except Exception as e:
        failed.append(pdb_id)
        print(f"Failed {pdb_id}: {e}")

df_cdr = pd.DataFrame(results)
df_cdr.to_csv('cdr_annotations.tsv', sep='\t', index=False)

print(f"Annotated: {len(df_cdr)}")
print(f"Failed: {len(failed)}")
if failed:
    print(f"Failed IDs: {failed}")
print(f"\nCDR-H3 length distribution:")
print(df_cdr['cdr_h3_len'].describe())
