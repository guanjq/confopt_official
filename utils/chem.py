import copy
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol, GetPeriodicTable

NODE_FEATS = {'aromatic', 'sp', 'sp2', 'sp3', 'num_hs'}
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def set_conformer_positions(conf, pos):
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    return conf


def update_data_rdmol_positions(data):
    for i in range(data.pos.size(0)):
        data.rdmol.GetConformer(0).SetAtomPosition(i, data.pos[i].tolist())
    return data


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = copy.deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def mol_to_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, allHsExplicit=True)


def remove_duplicate_mols(molecules: List[Mol]) -> List[Mol]:
    unique_tuples: List[Tuple[str, Mol]] = []

    for molecule in molecules:
        duplicate = False
        smiles = mol_to_smiles(molecule)
        for unique_smiles, _ in unique_tuples:
            if smiles == unique_smiles:
                duplicate = True
                break

        if not duplicate:
            unique_tuples.append((smiles, molecule))

    return [mol for smiles, mol in unique_tuples]


def get_atoms_in_ring(mol):
    atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            atoms.add(a)
    return atoms


def get_2D_mol(mol):
    mol = copy.deepcopy(mol)
    DP.Compute2DCoords(mol)
    return mol


def draw_mol_svg(mol, molSize=(450, 150), kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        DP.Compute2DCoords(mc)
    drawer = MD2.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    # return svg.replace('svg:','')
    return svg


def get_conjugated_features(mol):
    """Get conjugated features.
    Args:
        mol (rdMol): input rdkit mole with N atoms
    Returns:
        conj_grp (np.ndarray): (N,) shape numpy array
            indicating the Conjugated Group Index, and -1 if the value is not in any conjugated group
        same_conj_grp (np.ndarray): (N, N) shape numpy array
            indicating whether two atoms are in the same group. 0 for no, and 1 for yes.
    """

    # build resonance molecules for getting conjugated group
    resonance_mols = Chem.rdchem.ResonanceMolSupplier(mol)

    # order conj_grp index by atom index, same as the atom features and edge features
    conj_grp = []
    for atom_idx in range(mol.GetNumAtoms()):
        conjGrpIdx = Chem.rdchem.ResonanceMolSupplier.GetAtomConjGrpIdx(resonance_mols, atom_idx)
        # to get around a bug in rdkit where the c++ code return unsigned int for -1 value
        if conjGrpIdx >= np.iinfo(np.uint32).max:
            conjGrpIdx = np.iinfo(np.uint32).max - 1 - conjGrpIdx
        conj_grp.append(conjGrpIdx)
    conj_grp = np.array(conj_grp)

    # build same_conj_grp
    same_conj_grp = np.expand_dims(conj_grp, 0) == np.expand_dims(conj_grp, 1)
    real_conj_grp = conj_grp >= 0
    # 1 only for atom pair with real conjugated group
    real_conj_grp = np.expand_dims(real_conj_grp, 0) * np.expand_dims(real_conj_grp, 1)
    same_conj_grp = same_conj_grp * real_conj_grp
    same_conj_grp = same_conj_grp.astype(int)

    return conj_grp, same_conj_grp


def get_rotatable_bond(mol):
    """Get rotatable bond features.
        Args:
            mol (rdMol): input rdkit mole with N atoms
        Returns:
            r_bonds (np.ndarray): (N, N) shape numpy array indicating whether the bonds between two atoms are rotatable.
    """
    n = mol.GetNumAtoms()
    r_bonds = np.zeros((n, n))
    rot_atom_pairs = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    for i, j in rot_atom_pairs:
        r_bonds[i, j] = 1
        r_bonds[j, i] = 1
    return r_bonds


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies
