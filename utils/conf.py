import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, AllChem
from rdkit.Chem.rdchem import Mol


def add_conformer(mol: Mol, pos=None, other_mol=None):
    if pos is not None:
        # pos: [num_samples, N, 3]
        for sample_pos in pos:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, sample_pos[i])
            mol.AddConformer(conf, assignId=True)
    elif other_mol is not None:
        for conf in other_mol.GetConformers():
            mol.AddConformer(conf, assignId=True)
    return mol


def _get_rmsd(mol, pos, heavy_only):
    mol1 = copy.deepcopy(mol)
    mol2 = copy.deepcopy(mol)
    mol2.RemoveAllConformers()
    add_conformer(mol2, np.expand_dims(pos, 0))
    if heavy_only:
        rms1 = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol1), Chem.RemoveHs(mol2))
    else:
        rms1 = rdMolAlign.GetBestRMS(mol1, mol2)

    # consider the chiral case
    mirror_pos = copy.deepcopy(pos)
    mirror_pos[:, -1] *= -1
    mol2.RemoveAllConformers()
    add_conformer(mol2, np.expand_dims(mirror_pos, 0))
    if heavy_only:
        rms2 = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol1), Chem.RemoveHs(mol2))
    else:
        rms2 = rdMolAlign.GetBestRMS(mol1, mol2)

    return min(rms1, rms2)


def align_conformer(ref_mol: Mol, gen_mol: Mol, heavy_only=True, delta=0.5, quick_mode=True):
    n_cov = 0
    min_rms_list = []
    neg_gen_flag = [True] * len(gen_mol.GetConformers())
    for ref_conf in ref_mol.GetConformers():
        # construct a new mol, first conf: ref_conf, others: gen_conf
        if quick_mode:
            tmp_mol = copy.deepcopy(ref_mol)
            tmp_mol.RemoveAllConformers()
            tmp_mol.AddConformer(ref_conf, assignId=True)
            for gen_conf in gen_mol.GetConformers():
                tmp_mol.AddConformer(gen_conf, assignId=True)

            # align conformer
            atom_ids = []
            if heavy_only:
                atom_ids = [atom.GetIdx() for atom in tmp_mol.GetAtoms() if atom.GetAtomicNum() > 1]
            rms_list = []
            rdMolAlign.AlignMolConformers(tmp_mol, atomIds=atom_ids, RMSlist=rms_list)
        else:
            tmp_mol = copy.deepcopy(ref_mol)
            tmp_mol.RemoveAllConformers()
            tmp_mol.AddConformer(ref_conf)
            rms_list = []
            for gen_conf in gen_mol.GetConformers():
                rms_list.append(_get_rmsd(tmp_mol, gen_conf.GetPositions(), heavy_only))

        if len(rms_list) == 0:
            print('Warning: rms list is empty!')
            continue
        assert len(rms_list) == len(gen_mol.GetConformers())
        neg_gen_flag = np.where(np.asarray(rms_list) < delta, False, neg_gen_flag)

        min_rms = min(rms_list)
        # COV
        if min_rms < delta:
            n_cov += 1
        # MAT
        min_rms_list.append(min_rms)

    return n_cov / len(ref_mol.GetConformers()), sum(min_rms_list) / len(min_rms_list), sum(neg_gen_flag) / len(
        gen_mol.GetConformers()), min(min_rms_list)


def rdkit_addconf(mol, num_confs=1, align=False, ff_opt=False, seed=42):
    """
    Convert SMILES to rdkit.Mol with 3D coordinates
    """

    # mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # mol = Chem.AddHs(mol)
        # if num_confs > 1:
        cids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            randomSeed=seed,
            numThreads=0,
            maxAttempts=0,
            ignoreSmoothingFailures=True,
        )
        if len(cids) == 0:
            # print('Embed Molecule Fail')
            return None
        if align:
            rmslist = []
            AllChem.AlignMolConformers(mol, RMSlist=rmslist)
        if ff_opt:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=200)
        # else:
        #     if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        #         # print('Embed Molecule Fail')
        #         return None
        #     if ff_opt:
        #         AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        print('Fail to convert to RDKit molecule')
        return None


def cal_rdkit_pos(rdmol, num_confs=10, ff_opt=True, heavy_only=True, seed=42):
    rdkit_mol = copy.deepcopy(rdmol)
    if heavy_only:
        rdkit_mol = Chem.AddHs(rdkit_mol)
    rdkit_mol.RemoveAllConformers()
    try:
        # Chem.SanitizeMol(rdkit_mol)
        result = rdkit_addconf(rdkit_mol, num_confs=num_confs, ff_opt=ff_opt, seed=seed)
    except Chem.rdchem.AtomValenceException:
        result = None
        rdkit_mol.RemoveAllConformers()
    if heavy_only:
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
    N = rdkit_mol.GetNumAtoms()
    success = 1
    if result is None:
        rdkit_pos = np.random.rand(num_confs, N, 3)
        add_conformer(rdkit_mol, rdkit_pos)
        try:
            if ff_opt:
                AllChem.MMFFOptimizeMoleculeConfs(rdkit_mol, numThreads=0)
        except:
            rdkit_mol.RemoveAllConformers()
            rdkit_pos = np.random.rand(num_confs, N, 3)
            add_conformer(rdkit_mol, rdkit_pos)
            success = 0
    num_gen_confs = len(rdkit_mol.GetConformers())
    if num_gen_confs < num_confs:
        rdkit_pos = [rdkit_mol.GetConformers()[i].GetPositions() for i in range(num_gen_confs)]
        random_pos = [np.random.rand(N, 3) for _ in range(num_confs - num_gen_confs)]
        try:
            if ff_opt:
                random_mol = copy.deepcopy(rdkit_mol)
                random_mol.RemoveAllConformers()
                add_conformer(random_mol, random_pos)
                AllChem.MMFFOptimizeMoleculeConfs(random_mol, numThreads=0)
                random_pos = [random_mol.GetConformers()[i].GetPositions() for i in range(num_confs - num_gen_confs)]
        except:
            random_pos = [np.random.rand(N, 3) for _ in range(num_confs - num_gen_confs)]
        rdkit_pos = np.array(rdkit_pos + random_pos)
    else:
        rdkit_pos = [rdkit_mol.GetConformers()[i].GetPositions() for i in range(num_confs)]
        rdkit_pos = np.array(rdkit_pos)
    assert rdkit_pos.shape == (num_confs, N, 3)
    return rdkit_pos, success
