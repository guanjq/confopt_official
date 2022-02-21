import numpy as np
from tqdm.auto import tqdm

import copy
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdmolops import RemoveHs
# from confgf import utils


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    assert mol.GetNumAtoms() == pos.shape[0]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.AddConformer(conf, assignId=True)

    # for i in range(pos.shape[0]):
    #     mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = copy.deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd


def get_rmsd_confusion_matrix(rdmol, pos_ref, pos_gen, useFF=False):
    num_gen = pos_gen.size(0)
    num_ref = pos_ref.size(0)
    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen], dtype=np.float)

    for i in range(num_gen):
        gen_mol = set_rdmol_positions(rdmol, pos_gen[i])
        if useFF:
            # print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = set_rdmol_positions(rdmol, pos_ref[j])
            rmsd_confusion_mat[j, i] = GetBestRMSD(gen_mol, ref_mol)

    return rmsd_confusion_mat


def evaluate_conf(rdmol, pos_ref, pos_gen, useFF=False, swap=False, threshold=0.5):
    if swap:
        rmsd_confusion_mat = get_rmsd_confusion_matrix(rdmol, pos_gen, pos_ref, useFF=useFF)
    else:
        rmsd_confusion_mat = get_rmsd_confusion_matrix(rdmol, pos_ref, pos_gen, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    rmsd_gen_min = rmsd_confusion_mat.min(0)
    return (rmsd_ref_min <= threshold).mean(), rmsd_ref_min.mean(), (rmsd_gen_min > threshold).mean()
    # return (rmsd_ref_min <= threshold), rmsd_ref_min