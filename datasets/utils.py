import copy

import dgl
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType
from torch.nn import functional as F
from torch_scatter import scatter

from utils import chem as utils_chem


def rdmol_to_dict(mol: Mol, gt_smiles=None, conf_weights=None, heavy_only=True):
    ori_mol = copy.deepcopy(mol)
    if heavy_only:
        mol = Chem.RemoveHs(mol)
    N = mol.GetNumAtoms()
    E = mol.GetNumBonds()
    pos = torch.stack([torch.tensor(mol.GetConformer(i).GetPositions(), dtype=torch.float32)
                       for i in range(mol.GetNumConformers())])

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom_idx in range(N):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    node_type = torch.tensor(atomic_number, dtype=torch.long)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils_chem.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    node_feat = torch.tensor([aromatic, sp, sp2, sp3, num_hs], dtype=torch.float).t().contiguous()

    smiles = Chem.MolToSmiles(mol) if gt_smiles is None else gt_smiles
    if conf_weights is not None:
        conf_weights = torch.from_numpy(conf_weights)
    data = {'smiles': smiles, 'rdmol': ori_mol,
            'num_atoms': N, 'pos': pos, 'node_type': node_type, 'node_feat': node_feat,
            'num_bonds': E, 'edge_index': edge_index, 'edge_type': edge_type, 'conf_weights': conf_weights}
    return data


def dict_to_dgl_graph(data, edge_transform, node_remap, remap_types):
    N = data['num_atoms']
    # Load node features
    node_type = data['node_type']
    node_feat = data['node_feat']

    # Load edge features
    if edge_transform is not None:
        edge_index, edge_type, edge_order = edge_transform(data)
    else:
        edge_index, edge_type = data['edge_index'], data['edge_type']
        edge_order = torch.ones_like(edge_type)

    # Create graph
    src, dst = edge_index[0], edge_index[1]
    G = dgl.graph(data=(src, dst), num_nodes=N)

    # Add node features to graph
    G.ndata['node_type'] = node_type
    if node_remap is not None:
        remap_node_type = node_remap[node_type]
        G.ndata['remap_node_type'] = F.one_hot(remap_node_type, len(remap_types))
    pt = Chem.GetPeriodicTable()
    G.ndata['radius'] = torch.tensor(
        [pt.GetRcovalent(atomic_number) for atomic_number in node_type.tolist()]).view(-1, 1)
    G.ndata['node_feat'] = node_feat

    # Add edge features to graph
    G.edata['edge_type'] = edge_type
    G.edata['edge_order'] = edge_order
    return G
