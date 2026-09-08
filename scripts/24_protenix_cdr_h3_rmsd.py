import gemmi
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1
import numpy as np
import pandas as pd
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

DRIVE_PROT = 'gdrive:antibody_benchmark/protenix_outputs'
TMP = '/tmp/protenix_rmsd'
OUTFILE = 'protenix_cdr_h3_rmsd.tsv'
SEEDS = [101, 102, 103, 104, 105]
os.makedirs(TMP, exist_ok=True)

parser = PDBParser(QUIET=True)
df = pd.read_csv('benchmark_master.tsv', sep='\t')

def find_cdr_h3_ca_atoms(structure, chain_id, cdr_h3_seq):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
                full_seq = ''.join([seq1(r.resname, undef_code='X') for r in residues])
                idx = full_seq.find(cdr_h3_seq)
                if idx == -1:
                    return None, None
                h3_residues = residues[idx:idx+len(cdr_h3_seq)]
                ca_atoms = [r['CA'] for r in h3_residues if 'CA' in r]
                return h3_residues, ca_atoms
    return None, None

def get_framework_ca_atoms(structure, chain_id, cdr_h3_seq):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
                full_seq = ''.join([seq1(r.resname, undef_code='X') for r in residues])
                idx = full_seq.find(cdr_h3_seq)
                framework = residues[:idx] + residues[idx+len(cdr_h3_seq):] if idx != -1 else residues
                return [r['CA'] for r in framework if 'CA' in r]
    return []

if os.path.exists(OUTFILE):
    df_existing = pd.read_csv(OUTFILE, sep='\t')
    done = set(zip(df_existing['pdb'], df_existing['seed']))
    results = df_existing.to_dict('records')
    print(f"Resuming: {len(results)} done")
else:
    done = set()
    results = []

failed = []
total = len(results)

print(f"Starting Protenix CDR-H3 RMSD for {len(df)} complexes...")

for idx, (_, row) in enumerate(df.iterrows()):
    pdb_id = row['pdb']
    cdr_h3_seq = row['cdr_h3']
    hchain = row['Hchain']

    if not isinstance(cdr_h3_seq, str):
        continue

    native_path = f'pdb_structures/{pdb_id}.pdb'
    if not os.path.exists(native_path):
        continue

    if all((pdb_id, seed) in done for seed in SEEDS):
        continue

    try:
        native = parser.get_structure(pdb_id, native_path)
        native_fw = get_framework_ca_atoms(native, hchain, cdr_h3_seq)
        _, native_h3_ca = find_cdr_h3_ca_atoms(native, hchain, cdr_h3_seq)
        if not native_h3_ca:
            continue
        native_h3_coords = np.array([a.get_vector().get_array() for a in native_h3_ca])
    except:
        continue

    for seed in SEEDS:
        if (pdb_id, seed) in done:
            continue

        cif_name = f'{pdb_id}_sample_0.cif'
        cif_dst = f'{TMP}/{pdb_id}_seed{seed}.cif'
        pdb_dst = f'{TMP}/{pdb_id}_seed{seed}.pdb'

        try:
            if not os.path.exists(cif_dst):
                r = subprocess.run([
                    'rclone', 'copy',
                    f'{DRIVE_PROT}/{pdb_id}/seed_{seed}/{cif_name}',
                    TMP
                ], capture_output=True, timeout=120)
                src = f'{TMP}/{cif_name}'
                if os.path.exists(src):
                    os.rename(src, cif_dst)

            if not os.path.exists(cif_dst):
                failed.append((pdb_id, seed))
                continue

            doc = gemmi.cif.read(cif_dst)
            gemmi.make_structure_from_block(doc.sole_block()).write_pdb(pdb_dst)
            pred = parser.get_structure('pred', pdb_dst)

            pred_fw = get_framework_ca_atoms(pred, hchain, cdr_h3_seq)
            min_fw = min(len(native_fw), len(pred_fw))
            if min_fw < 10:
                continue

            sup = Superimposer()
            sup.set_atoms(native_fw[:min_fw], pred_fw[:min_fw])
            sup.apply(list(pred[0].get_atoms()))

            _, pred_h3_ca = find_cdr_h3_ca_atoms(pred, hchain, cdr_h3_seq)
            if not pred_h3_ca:
                continue

            pred_h3_coords = np.array([a.get_vector().get_array() for a in pred_h3_ca])
            min_h3 = min(len(native_h3_coords), len(pred_h3_coords))

            cdr_h3_rmsd = np.sqrt(np.mean(np.sum(
                (native_h3_coords[:min_h3] - pred_h3_coords[:min_h3])**2, axis=1)))

            results.append({
                'pdb': pdb_id,
                'seed': seed,
                'cdr_h3_rmsd': round(cdr_h3_rmsd, 4),
                'fw_rmsd': round(sup.rms, 4),
                'cdr_h3_len': len(cdr_h3_seq),
                'has_light': pd.notna(row['lseq'])
            })
            done.add((pdb_id, seed))
            total += 1

            if os.path.exists(pdb_dst):
                os.remove(pdb_dst)

        except Exception as e:
            failed.append((pdb_id, seed))

    if (idx + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
        print(f"[{idx+1}/{len(df)}] Computed: {total}, Failed: {len(failed)}")

pd.DataFrame(results).to_csv(OUTFILE, sep='\t', index=False)
print(f"\nDone. Computed: {len(results)}, Failed: {len(failed)}")
df_out = pd.DataFrame(results)
print(f"\nCDR-H3 RMSD stats (Protenix):")
print(df_out['cdr_h3_rmsd'].describe())
print(f"\nFramework RMSD stats:")
print(df_out['fw_rmsd'].describe())
