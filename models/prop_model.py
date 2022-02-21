import ase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets.qm9 import atomrefs
from torch_geometric.nn import SchNet, NNConv, Set2Set
from torch_scatter import scatter

from models import utils
from models.baseline_models.egnn import EnEquiNetwork
from models.utils import MLP
from utils.chem import NODE_FEATS, BOND_TYPES
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def atomref(target):
    if target in atomrefs:
        out = torch.zeros(100)
        out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
        return out.view(-1, 1)
    return None


def get_prop_model(model_type, config, mean, std, target_idx, aux_edge_order):
    if model_type == 'schnet':
        net = SchNetPropPred(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_interactions,
            num_gaussians=config.num_gaussians, cutoff=config.cutoff,
            mean=mean, std=std,
            dipole=True if target_idx == 0 else False,
            atomref=atomref(target_idx)
        )
    elif model_type == 'mpnn':
        net = MPNNPropPred(
            node_dim=100 + len(NODE_FEATS),
            edge_dim=len(BOND_TYPES) + aux_edge_order,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            mp_iter=config.mp_iter,
            set2set_iter=config.set2set_iter,
            mean=mean, std=std,
        )
    elif model_type == 'egnn':
        net = EnPropPred(
            num_layers=config.num_layers,
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            output_dim=1,
            act_fn=config.act_fn,
            norm=config.norm,
            update_x=config.update_x,
            mean=mean, std=std,
            max_charge=config.max_charge,
            dipole=True if target_idx == 0 else False,
            atomref=atomref(target_idx)
        )

    elif model_type == 'ours_o2':
        net = Uni2TwoUpPropPred(
            num_layers=config.num_layers,
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            output_dim=1,
            n_heads=config.n_heads,
            r_feat_mode=config.r_feat_mode,
            energy_h_mode=config.energy_h_mode,
            ew_net_type=config.ew_net_type,
            update_x=config.update_x,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            mean=mean, std=std,
            max_charge=config.max_charge,
            dipole=True if target_idx == 0 else False,
            atomref=atomref(target_idx)
        )
    else:
        raise ValueError(model_type)
    return net


class PropPredNet(nn.Module):
    def __init__(self, config, aux_edge_order=1, target_idx=0, target_mean=None, target_std=None):
        super(PropPredNet, self).__init__()
        self.model_type = config.model_type
        self.mean = target_mean
        self.std = target_std
        self.register_buffer('target_mean', target_mean)
        self.register_buffer('target_std', target_std)
        self.net = get_prop_model(self.model_type, config, self.mean, self.std, target_idx, aux_edge_order)

    def forward(self, G, pos_type, fix_init_pos=None):
        if fix_init_pos is not None:
            pos = fix_init_pos
        else:
            if pos_type == 'gt':
                pos = G.ndata['pos']
            elif pos_type == 'rdkit':
                pos = G.ndata['rdkit_pos']
            elif pos_type == 'pre':
                pos = G.ndata['model_pos']
            else:
                raise ValueError(f'Invalid pos type: {pos_type}')

        batch_idx = utils.convert_dgl_to_batch(G, pos.device)
        if self.model_type == 'schnet':
            out = self.net(G.ndata['node_type'], pos, batch_idx)
            return out, None
        elif self.model_type == 'mpnn':
            out = self.net(G, batch_idx)
            return out, None
        elif self.model_type in ['egnn', 'ours_o2']:
            out, x = self.net(G, pos, batch_idx, pos_type)
            return out, x
        else:
            raise ValueError(f'Invalid model type: {self.model_type}')


class SchNetPropPred(SchNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        delattr(self, "atomic_mass")
        atomic_mass = torch.from_numpy(ase.data.atomic_masses.astype(np.float32))
        self.register_buffer('atomic_mass', atomic_mass)


class MPNNPropPred(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, mp_iter=3, set2set_iter=4, mean=None, std=None):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        # self.edge_emb = nn.Embedding(edge_dim, hidden_dim)  # len(BOND_TYPES) + aux_order
        self.mp_iter = mp_iter
        self.set2set_iter = set2set_iter
        self.mpnn_blocks = []
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        for i in range(mp_iter):
            conv = NNConv(hidden_dim, hidden_dim, nn=self.edge_net)
            self.mpnn_blocks.append(conv)
        self.mpnn_blocks = nn.ModuleList(self.mpnn_blocks)
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_iter)
        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mean = mean
        self.std = std

    def forward(self, G, batch):
        edge_index = torch.stack(G.edges()).to(G.ndata['node_type'].device)
        node_feat = torch.cat([F.one_hot(G.ndata['node_type'], 100).to(G.ndata['node_feat']),
                               G.ndata['node_feat']], dim=-1)
        h = self.node_emb(node_feat)
        edge_attr = F.one_hot(G.edata['edge_type'], self.edge_dim).to(G.ndata['node_feat'])

        for i in range(self.mp_iter):
            h = self.mpnn_blocks[i](h, edge_index, edge_attr)
        pre = self.set2set(h, batch)
        out = self.output_block(pre)

        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean
        return out


class QM9PropPredModule(nn.Module):
    def forward(self, G, pos, batch, pos_type='gt'):
        charge = G.ndata['node_type'].to(pos)
        charge_feat = torch.stack([(charge / self.max_charge) ** power for power in range(0, 3)], dim=1)  # [N, 3]
        node_feat = (G.ndata['remap_node_type'].to(pos).unsqueeze(-1) * charge_feat.unsqueeze(1)).view(pos.shape[0], -1)
        G.ndata['x'] = pos
        G.ndata['f'] = self.node_emb(node_feat).unsqueeze(-1)
        if self.edge_dim > 0:
            G.edata['w'] = self.edge_emb(G.edata['edge_type'])
        x, _, all_h = self.net(G, return_h=True)
        h = all_h[-1]
        h = self.pre_out_block(h)

        # output part adapted from SchNet
        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[G.ndata['node_type']].view(-1, 1)
            atom_pos = pos if pos_type == 'gt' else x
            c = scatter(mass * atom_pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (atom_pos - c[batch])

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(G.ndata['node_type'])

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if not self.dipole and self.mean is not None and self.std is not None:
            out = out * self.std + self.mean

        return out, x


class EnPropPred(QM9PropPredModule):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, num_r_gaussian, output_dim,
                 act_fn='relu', norm=False, update_x=False,
                 mean=None, std=None, readout='sum', max_charge=9, dipole=False, atomref=None):
        super().__init__()
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.output_dim = output_dim
        self.act_fn = act_fn
        self.norm = norm
        self.update_x = update_x
        self.max_charge = max_charge

        self.node_emb = nn.Linear(node_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_emb = nn.Embedding(edge_dim, hidden_dim)  # len(BOND_TYPES) + aux_order
        self.net = EnEquiNetwork(
            num_layers, hidden_dim, edge_dim, num_r_gaussian, update_x=update_x, act_fn=act_fn, norm=norm)

        self.pre_out_block = MLP(hidden_dim, output_dim, hidden_dim, num_layer=2, norm=False, act_fn=act_fn)
        self.mean = mean
        self.std = std
        # output related
        self.dipole = dipole
        self.readout = 'add' if self.dipole else readout
        atomic_mass = torch.from_numpy(ase.data.atomic_masses.astype(np.float32))
        self.register_buffer('atomic_mass', atomic_mass)
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)


class Uni2TwoUpPropPred(QM9PropPredModule):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim,
                 num_r_gaussian=50, num_node_types=5,  output_dim=1, n_heads=1,
                 r_feat_mode='basic', energy_h_mode='basic', ew_net_type='none', update_x=False,
                 act_fn='relu', norm=False, cutoff_mode='none',
                 mean=None, std=None, readout='sum', max_charge=9, dipole=False, atomref=None):
        super().__init__()
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.num_node_types = num_node_types
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.r_feat_mode = r_feat_mode
        self.energy_h_mode = energy_h_mode
        self.ew_net_type = ew_net_type
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.max_charge = max_charge

        if 'mix' in self.energy_h_mode:
            self.node_emb = nn.Linear(node_dim, hidden_dim * 2)
        else:
            self.node_emb = nn.Linear(node_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_emb = nn.Embedding(edge_dim, hidden_dim)  # len(BOND_TYPES) + aux_order
        self.net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=1, num_layers=num_layers, hidden_dim=hidden_dim, n_heads=n_heads,
            num_r_gaussian=num_r_gaussian, edge_feat_dim=edge_dim, num_node_types=num_node_types,
            act_fn=act_fn, norm=norm,
            cutoff_mode=cutoff_mode, ew_net_type=ew_net_type, r_feat_mode=r_feat_mode, energy_h_mode=energy_h_mode,
            num_init_x2h=0, num_init_h2x=0, num_x2h=1, num_h2x=1 if update_x else 0,
            r_max=10., x2h_out_fc=True, sync_twoup=False
        )

        hout_dim = hidden_dim * 2 if 'mix' in energy_h_mode else hidden_dim
        self.pre_out_block = MLP(hout_dim, output_dim, hidden_dim, num_layer=2, norm=False, act_fn=act_fn)
        self.mean = mean
        self.std = std
        # output related
        self.dipole = dipole
        self.readout = 'add' if self.dipole else readout
        atomic_mass = torch.from_numpy(ase.data.atomic_masses.astype(np.float32))
        self.register_buffer('atomic_mass', atomic_mass)
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)
