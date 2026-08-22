"""
Script 9: Extract antigen contact positions from CDR-H3
Input: benchmark_master.tsv, pdb_structures/
Output: boltz2_contact_positions.tsv

For each complex, identifies which CDR-H3 residues are within
4.5 Angstrom of any antigen atom in the experimental structure.
These are antigen contact positions — where binding specificity lives.

Key finding: Mean contact fraction = 46.5% of CDR-H3 residues
contact the antigen. These positions have amino acid recovery
of only 8-23% in co-folding models (vs 23-51% at non-contact).

This supports Angle 2 (contact position density) of the
mechanistic framework.

Requirements: pip install biopython
"""

import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings
warnings.filterwarnings('ignore')

def get_contact_positions(native, hchain, antigen_chains, cdr_h3_seq, cutoff=4.5):
    """Find CDR-H3 residues within cutoff Angstrom of any antigen atom."""
    cdr_h3_residues = None
    for model in native:
        for chain in model:
            if chain.id == hchain:
                residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
                full_seq = ''.join([seq1(r.resname, undef_code='X') for r in residues])
                idx = full_seq.find(cdr_h3_seq)
                if idx == -1:
                    return None
                cdr_h3_residues = residues[idx:idx+len(cdr_h3_seq)]
                break

    if cdr_h3_residues is None:
        return None

    antigen_atoms = []
    for model in native:
        for chain in model:
            if chain.id in antigen_chains:
                for residue in chain.get_residues():
                    if residue.get_id()[0] == ' ':
                        antigen_atoms.extend(list(residue.get_atoms()))

    if not antigen_atoms:
        return None

    contact_flags = []
    for res in cdr_h3_residues:
        is_contact = False
        for atom in res.get_atoms():
            for ag_atom in antigen_atoms:
                if atom - ag_atom < cutoff:
                    is_contact = True
                    break
            if is_contact:
                break
        contact_flags.append(is_contact)

    return contact_flags

parser = PDBParser(QUIET=True)
df = pd.read_csv('benchmark_master.tsv', sep='\t')

results = []
failed = []

print(f"Extracting contact positions for {len(df)} complexes...")

for i, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    hchain = row['Hchain']
    cdr_h3_seq = row['cdr_h3']
    antigen_chains = [c.strip() for c in str(row['antigen_chain']).split('|')]

    if not isinstance(cdr_h3_seq, str):
        failed.append(pdb_id)
        continue

    native_path = f'pdb_structures/{pdb_id}.pdb'
    if not os.path.exists(native_path):
        failed.append(pdb_id)
        continue

    try:
        native = parser.get_structure(pdb_id, native_path)
        contact_flags = get_contact_positions(
            native, hchain, antigen_chains, cdr_h3_seq, cutoff=4.5)

        if contact_flags is None:
            failed.append(pdb_id)
            continue

        n_contact = sum(contact_flags)
        n_total = len(contact_flags)

        results.append({
            'pdb': pdb_id,
            'cdr_h3_len': n_total,
            'n_contact': n_contact,
            'n_noncontact': n_total - n_contact,
            'contact_fraction': n_contact / n_total if n_total > 0 else 0,
            'contact_flags': ','.join(map(str, contact_flags))
        })

        if (i+1) % 50 == 0:
            print(f"  {i+1}/304 done...")

    except Exception as e:
        failed.append(pdb_id)
        print(f"Failed {pdb_id}: {e}")

df_contacts = pd.DataFrame(results)
df_contacts.to_csv('boltz2_contact_positions.tsv', sep='\t', index=False)

print(f"\nDone. Computed: {len(results)}, Failed: {len(failed)}")
print(f"\nContact position stats:")
print(df_contacts[['cdr_h3_len', 'n_contact', 'contact_fraction']].describe())
