import os
import pickle
import random

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from datasets.utils import rdmol_to_dict, dict_to_dgl_graph
from utils import chem as utils_chem
from utils.conf import cal_rdkit_pos
from utils.transforms import get_higher_order_adj_matrix


class ConfDatasetDGL(Dataset):
    def __init__(self, raw_path, force_reload=False, heavy_only=False, rdkit_mol=True,
                 rdkit_pos_mode='random', n_ref_samples=1, n_gen_samples=1,
                 edge_transform=None, processed_tag=None, size_limit=None, seed=2020, mode='lowest',
                 lowest_thres=1.0):
        super().__init__()
        assert rdkit_pos_mode in ['random', 'none', 'all', 'multi', 'online', 'online_ff']
        assert mode in ['lowest', 'relax_lowest', 'random_low', 'multi_low', 'multi_sample_low']
        self.raw_path = raw_path
        prefix = []
        if heavy_only:
            prefix.append('heavy')
        if rdkit_mol:
            prefix.append('rdkit')
        if processed_tag is not None:
            prefix.append(processed_tag)
        self.processed_pkl = raw_path + '.' + '_'.join(prefix)
        self.processed_dir = raw_path + '.cache.dir'  # index mode, save memory
        self.index_mode = False
        self.heavy_only = heavy_only
        self.rdkit_mol = rdkit_mol
        self.size_limit = size_limit
        self.seed = seed
        self.mode = mode
        self.rdkit_pos_mode = rdkit_pos_mode
        self.n_ref_samples = n_ref_samples  # valid for multi_low mode and online rdkit is True
        self.n_gen_samples = n_gen_samples
        self.node_remap = torch.zeros(100, dtype=torch.long)  # to remap node type to one-hot vector
        self.remap_types = ['UNK', 6, 7, 8, 9, 16, 17, 35]  # common node types in Drug (and QM9)
        for idx, t in enumerate(self.remap_types[1:]):
            self.node_remap[t] = idx + 1
        self.lowest_thres = lowest_thres

        self.dataset = None
        if force_reload or not (os.path.exists(self.processed_pkl) or os.path.exists(self.processed_dir)):
            self.process_mols()
        elif os.path.exists(self.processed_dir):
            print(f'Loading {self.raw_path} in index mode.')
            self.index_mode = True
            data_index = torch.load(os.path.join(self.processed_dir, 'index.pt'))
            self.cache_dir = data_index['data_dir']
            self.cache_files = np.array(data_index['files'])
        else:
            self.load_processed()

        self.edge_transform = edge_transform

    @staticmethod
    def process_single_data_point(data, include_rdkit_mol=True, heavy_only=True):
        if isinstance(data, Mol):
            smiles, conf_weights = None, None
            mol = data
        else:
            smiles, mol, conf_weights = data
        try:
            data = rdmol_to_dict(mol, smiles, conf_weights, heavy_only=heavy_only)
        except:
            print(f'Convert Mol to Data Fail!')
            return None
        if include_rdkit_mol:
            data['rdkit_pos'], data['rdkit_success'] = cal_rdkit_pos(mol, num_confs=10,
                                                                     ff_opt=True,
                                                                     heavy_only=heavy_only, seed=-1)
        return data

    def process_mols(self):
        self.dataset = []
        with open(self.raw_path, 'rb') as f:
            mols_db = pickle.load(f)
        n, n_fail = 0, 0
        for data in tqdm(mols_db):
            n += 1
            data = self.process_single_data_point(data, self.rdkit_mol, self.heavy_only)
            if data is None:
                n_fail += 1
                continue
            self.dataset.append(data)
        print(f'{n} mols in total, successfully convert {len(self.dataset)}, fail {n_fail}')
        torch.save({
            'dataset': self.dataset,
        }, self.processed_pkl)
        if self.size_limit is not None and self.size_limit > 0:
            random.Random(self.seed).shuffle(self.dataset)
            self.dataset = self.dataset[:self.size_limit]

    def load_processed(self):
        print(f'Load existing processed file from {self.processed_pkl}')
        self.dataset = torch.load(self.processed_pkl)['dataset']
        if self.size_limit is not None and self.size_limit > 0:
            random.Random(self.seed).shuffle(self.dataset)
            self.dataset = self.dataset[:self.size_limit]

    def __len__(self):
        if self.index_mode:
            return len(self.cache_files)
        else:
            return len(self.dataset)

    def get_label(self, data):
        if self.mode == 'lowest':  # conf with lowest conf, for the optimization task
            labels = data['pos'] if len(data['pos'].shape) == 2 else data['pos'][0]
            if 'conf_weights' in data:
                conf_prob = data['conf_weights'][0]
            else:
                conf_prob = [1]

        elif self.mode == 'relax_lowest':
            if 'conf_weights' in data and data['conf_weights'] is not None:
                conf_weights = np.array(data['conf_weights'])
            else:  # single gt pos
                conf_weights = np.array([1.0])
            prob = conf_weights / np.sum(conf_weights)
            conf_prob = prob[prob >= prob[0] * self.lowest_thres]
            labels = data['pos'][:len(conf_prob)]

        elif self.mode == 'random_low':
            assert len(data['pos'].shape) == 3
            rand_idx = random.randint(0, len(data['pos']) - 1)
            labels = data['pos'][rand_idx]
            conf_prob = data['conf_weights'][rand_idx]

        elif self.mode == 'multi_low':
            assert len(data['pos'].shape) == 3
            labels = data['pos']
            conf_prob = data['conf_weights']

        elif self.mode == 'multi_sample_low':
            assert len(data['pos'].shape) == 3
            assert len(data['pos']) == len(data['conf_weights'])
            # prob = data['conf_weights'] / data['conf_weights'].sum()
            # sample_id = np.random.choice(np.arange(len(data['pos'])), self.n_ref_samples, p=prob)
            sample_id = np.random.choice(np.arange(len(data['pos'])), self.n_ref_samples)
            labels = data['pos'][sample_id]
            conf_prob = data['conf_weights'][sample_id]

        else:
            raise ValueError(self.mode)
        return labels, conf_prob

    def __getitem__(self, idx):
        if self.index_mode:
            data = torch.load(os.path.join(self.cache_dir, self.cache_files[idx]))
        else:
            data = self.dataset[idx]

        G = dict_to_dgl_graph(data, self.edge_transform, self.node_remap, self.remap_types)

        N = G.num_nodes()
        src, dst = G.edges()
        # compute conjugated features
        rdmol = data['rdmol'] if not self.heavy_only else Chem.RemoveHs(data['rdmol'])
        conj_grp, same_conj_grp = utils_chem.get_conjugated_features(rdmol)
        conj_features = same_conj_grp[src, dst]
        G.edata['edge_feat'] = torch.tensor(conj_features).to(dtype=torch.float).view(-1, 1)

        # compute rotatable bonds
        r_bonds = utils_chem.get_rotatable_bond(rdmol)
        rotatable_bond = r_bonds[src, dst]
        G.edata['rotatable_bond'] = torch.tensor(rotatable_bond).to(dtype=torch.float)

        # label: [num_ref_confs, num_nodes, 3] or [num_nodes, 3]
        labels, conf_prob = self.get_label(data)

        adj = to_dense_adj(data['edge_index'], max_num_nodes=N).squeeze(0)  # special case, such as 'N.N#C/C=C\CN=C=O'
        adj_order = get_higher_order_adj_matrix(adj, 10)  # (N, N)

        n_gen_samples = 2 * len(labels) if self.n_gen_samples == 'auto' else self.n_gen_samples
        if self.rdkit_pos_mode == 'random':
            rand_idx = random.randint(0, len(data['rdkit_pos']) - 1)
            G.ndata['rdkit_pos'] = torch.tensor(data['rdkit_pos'][rand_idx], dtype=torch.float32)
        elif self.rdkit_pos_mode == 'none':
            # G.ndata['rdkit_pos'] = torch.randn(N, 3, dtype=torch.float32)
            pass
        elif self.rdkit_pos_mode == 'all':
            G.ndata['rdkit_pos'] = torch.tensor(data['rdkit_pos'], dtype=torch.float32).permute(1, 0, 2)
        elif self.rdkit_pos_mode == 'multi':
            sample_id = np.random.choice(np.arange(len(data['rdkit_pos'])), n_gen_samples)
            G.ndata['rdkit_pos'] = torch.tensor(data['rdkit_pos'][sample_id], dtype=torch.float32).permute(1, 0, 2)
        elif 'online' in self.rdkit_pos_mode:
            rdkit_pos, _ = cal_rdkit_pos(data['rdmol'],
                                         num_confs=n_gen_samples,
                                         ff_opt=True if self.rdkit_pos_mode == 'online_ff' else False,
                                         heavy_only=self.heavy_only, seed=-1)
            if n_gen_samples == 1:
                G.ndata['rdkit_pos'] = torch.tensor(rdkit_pos[0], dtype=torch.float32)
            else:
                G.ndata['rdkit_pos'] = torch.tensor(rdkit_pos, dtype=torch.float32).permute(1, 0, 2)
        else:
            raise ValueError(self.rdkit_pos_mode)

        meta_info = {
            'idx': idx,
            'smiles': data['smiles'],
            'rdmol': rdmol,
            'ori_rdmol': data['rdmol'],
            'conf_weights': conf_prob,
            'adj_order': adj_order,
            'rdkit_success': data['rdkit_success'] if 'rdkit_success' in data.keys() else 1,
            'all_rdkit_pos': data['rdkit_pos'] if 'rdkit_pos' in data else None
        }

        return G, labels, meta_info
