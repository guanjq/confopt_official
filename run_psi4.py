N_CONF = 5
SEED = 7
RDMOL_FILE = 'logs/ours_o2/test_dumps.pkl'
OUR_FILE = 'dump_results/qm9_prop_test/model_ours_o2_gen_conf.pkl'
EGNN_FILE = 'dump_results/qm9_prop_test/model_egnn_gen_conf.pkl'
RDKIT_FILE = 'dump_results/qm9_prop_test/rdkit_gen_conf.pkl'
OUTPUT_FILE = 'psi4_opt_full.csv'

import copy
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdMolAlign
import psi4

psi4.core.be_quiet()
psi4.set_memory('32 GB')
psi4.set_num_threads(32)


def mol_with_pos(mol, pos):
    """Assign 3d coordinates to a mol."""

    pos = copy.deepcopy(pos)

    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i])
    mol.AddConformer(conf, assignId=True)

    pos[..., -1] *= -1
    mir_mol = copy.deepcopy(mol)
    mir_mol.RemoveAllConformers()
    conf = Chem.Conformer(mir_mol.GetNumAtoms())
    for i in range(mir_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i])
    mir_mol.AddConformer(conf, assignId=True)

    return mol, mir_mol


def compute_property(mol):
    xyz = rdmolfiles.MolToXYZBlock(mol)
    xyz = '\n'.join(xyz.split('\n')[1:])
    psi4.geometry(xyz)
    psi4.set_options({'reference': 'uhf'})
    _, scf_wfn = psi4.energy('MP2/6-311++G(d,p)', return_wfn=True)
    scf_h = scf_wfn.epsilon_a_subset("AO", "ALL").to_array()[scf_wfn.nalpha()]
    scf_l = scf_wfn.epsilon_a_subset("AO", "ALL").to_array()[scf_wfn.nalpha() + 1]
    return scf_h, scf_l


with open(RDMOL_FILE, 'rb') as f:
    datas = pickle.load(f)
with open(OUR_FILE, 'rb') as f:
    ours = pickle.load(f)
with open(EGNN_FILE, 'rb') as f:
    egnns = pickle.load(f)
with open(RDKIT_FILE, 'rb') as f:
    rdkits = pickle.load(f)

idxs = np.arange(len(datas))
np.random.RandomState(seed=SEED).shuffle(idxs)

df = []
item_count = 0
for idx in idxs:
    item_count += 1
    try:
        data = datas[idx]
        rdmol = data['mol']
        gt_mol, _ = mol_with_pos(rdmol, data['gt_pos'][0])
        gt_h, gt_l = compute_property(Chem.AddHs(gt_mol, addCoords=True))

        for conf_id in range(N_CONF):
            rdkit_mol, rdkit_mol_mir = mol_with_pos(rdmol, rdkits[idx][conf_id])
            our_mol, our_mol_mir = mol_with_pos(rdmol, ours[idx][conf_id])
            egnn_mol, egnn_mol_mir = mol_with_pos(rdmol, egnns[idx][conf_id])

            our_rmsd = min(rdMolAlign.GetBestRMS(our_mol, gt_mol), rdMolAlign.GetBestRMS(our_mol_mir, gt_mol))
            rdkit_rmsd = min(rdMolAlign.GetBestRMS(rdkit_mol, gt_mol), rdMolAlign.GetBestRMS(rdkit_mol_mir, gt_mol))
            egnn_rmsd = min(rdMolAlign.GetBestRMS(egnn_mol, gt_mol), rdMolAlign.GetBestRMS(egnn_mol_mir, gt_mol))

            rdkit_h, rdkit_l = compute_property(Chem.AddHs(rdkit_mol, addCoords=True))
            our_h, our_l = compute_property(Chem.AddHs(our_mol, addCoords=True))
            egnn_h, egnn_l = compute_property(Chem.AddHs(egnn_mol, addCoords=True))

            df.append({
                'idx': idx,
                'conf_id': conf_id,
                'gt_h': gt_h,
                'gt_l': gt_l,
                'rdkit_h': rdkit_h,
                'rdkit_l': rdkit_l,
                'our_h': our_h,
                'our_l': our_l,
                'egnn_h': egnn_h,
                'egnn_l': egnn_l,
                'our_rmsd': our_rmsd,
                'rdkit_rmsd': rdkit_rmsd,
                'egnn_rmsd': egnn_rmsd,
            })
    except Exception as e:
        print(f'Fail the calculation for {idx} due to {e}.')

    if item_count == 1 or item_count % 10 == 0:
        pd.DataFrame(df).to_csv(OUTPUT_FILE, index=False)
        print(f'Saving pd.DataFrame as .csv for a total of {item_count} items.')

pd.DataFrame(df).to_csv(OUTPUT_FILE, index=False)
