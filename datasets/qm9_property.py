import os
import pickle
import random

import ase.units as units
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.utils import rdmol_to_dict, dict_to_dgl_graph
from utils.conf import cal_rdkit_pos

HAR2EV = units.Hartree
KCALMOL2EV = units.kcal / units.mol
DEBYE2EV = units.Debye
BOHR2EV = units.Bohr

UNIT_CONVERSION = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

TARGET_NAMES = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'A', 'B', 'C']


class QM9PropertyDataset(Dataset):
    def __init__(self, raw_path, target_name=None, force_reload=False, heavy_only=False, rdkit_mol=True,
                 processed_tag='dgl_processed', edge_transform=None,
                 split_file=None, split_type='train', model_pos_path=None, seed=2020):
        super().__init__()
        self.raw_path = raw_path
        self.mol_path = os.path.join(raw_path, 'raw', 'gdb9.sdf')
        self.target_path = os.path.join(raw_path, 'raw', 'gdb9.sdf.csv')
        self.target_name = target_name
        self.target_idx = TARGET_NAMES.index(self.target_name)

        """
        | Target Name | Unit in raw | Unit reported
        | 'mu'        | D           | D
        | 'alpha'     | Bohr^3      | Bohr^3
        | 'homo'      | Hartree     | meV
        | 'lumo'      | Hartree     | meV
        | 'gap'       | Hartree     | meV
        | 'r2'        | Bohr^2      | Bohr^2
        | 'zpve'      | Hartree     | meV
        
        | 'u0'        | Hartree     | meV
        | 'u298'      | Hartree     | meV
        | 'h298'      | Hartree     | meV
        | 'g298'      | Hartree     | meV
        
        | 'u0_atom'   | kcal/mol    | meV
        | 'u298_atom' | kcal/mol    | meV
        | 'h298_atom' | kcal/mol    | meV
        | 'g298_atom' | kcal/mol    | meV
        
        | 'Cv'        | cal/(mol*K) | cal/(mol*K)
        """

        prefix = []
        if heavy_only:
            prefix.append('heavy')
        if rdkit_mol:
            prefix.append('rdkit')
        if processed_tag is not None:
            prefix.append(processed_tag)
        self.processed_path = os.path.join(raw_path, f'qm9_{split_type}' + '.' + '_'.join(prefix))
        self.heavy_only = heavy_only
        self.rdkit_mol = rdkit_mol
        self.split_type = split_type
        self.split_idx = (np.load(split_file)[split_type] - 1).tolist()
        self.node_remap = torch.zeros(100, dtype=torch.long)
        self.remap_types = [1, 6, 7, 8, 9]
        for idx, t in enumerate(self.remap_types):
            self.node_remap[t] = idx
        self.seed = seed

        self.dataset = None
        if force_reload or not os.path.exists(self.processed_path):
            self.process_mols()
        else:
            self.load_processed()

        self.edge_transform = edge_transform
        self.model_pos_path = model_pos_path
        if model_pos_path is not None:
            print('Load model pos from: ', model_pos_path)
            with open(model_pos_path, 'rb') as f:
                self.model_pos = pickle.load(f)
            assert len(self.model_pos) == len(self.dataset)

    def process_mols(self):
        self.dataset = []
        suppl = Chem.SDMolSupplier(self.mol_path, removeHs=False, sanitize=False)
        with open(self.target_path, 'r') as f:
            content = f.read().split('\n')
            target = [[float(x) for x in line.split(',')[1:20]] for line in content[1:-1]]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * UNIT_CONVERSION.view(1, -1)

        rdkit_fail = 0
        for i in tqdm(self.split_idx):
            mol = suppl[i]
            data = rdmol_to_dict(mol, heavy_only=self.heavy_only)
            data['y'] = target[i]
            if self.rdkit_mol:
                data['rdkit_pos'], data['rdkit_success'] = cal_rdkit_pos(mol, num_confs=10,
                                                                         ff_opt=True, heavy_only=self.heavy_only)
                rdkit_fail += 1 - data['rdkit_success']
            self.dataset.append(data)
        print(f'{len(self.split_idx)} mols in total, successfully convert {len(self.dataset)}, rdkit fail: {rdkit_fail}')
        processed_data = {'dataset': self.dataset}
        if self.split_type == 'train':
            target_mean = target[self.split_idx].mean(0)
            target_std = target[self.split_idx].std(0)
            processed_data.update({'target_mean': target_mean, 'target_std': target_std})
        torch.save(processed_data, self.processed_path)

    def load_processed(self):
        print(f'Load existing processed file from {self.processed_path}')
        processed_data = torch.load(self.processed_path)
        self.dataset = processed_data['dataset']
        if 'target_mean' in processed_data.keys():
            self.target_mean = processed_data['target_mean']
        if 'target_std' in processed_data.keys():
            self.target_std = processed_data['target_std']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        G = dict_to_dgl_graph(data, self.edge_transform, self.node_remap, self.remap_types)
        G.ndata['pos'] = data['pos'] if len(data['pos'].shape) == 2 else data['pos'][0]
        labels = data['y']

        meta_info = {'smiles': data['smiles'],
                     'rdmol': data['rdmol'] if not self.heavy_only else Chem.RemoveHs(data['rdmol']),
                     'ori_rdmol': data['rdmol']}
        if self.rdkit_mol:
            rand_idx = random.randint(0, len(data['rdkit_pos']) - 1)
            G.ndata['rdkit_pos'] = torch.tensor(data['rdkit_pos'][rand_idx], dtype=torch.float32)
            if 'rdkit_success' in data.keys():
                meta_info['rdkit_success'] = data['rdkit_success']
        if self.model_pos_path:
            # randomly choose a pre-computed pos
            model_pos = self.model_pos[idx]
            rand_idx = random.randint(0, len(model_pos) - 1)
            G.ndata['model_pos'] = torch.tensor(model_pos[rand_idx], dtype=torch.float32)

        return G, labels, meta_info
