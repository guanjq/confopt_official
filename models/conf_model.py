import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.chem import NODE_FEATS, BOND_TYPES
from models import utils
from models.baseline_models.egnn import EnEquiNetwork
from models.baseline_models.se3_trans import EquiSE3Transformer
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from models.uni_transformer_o3 import UniTransformerO3TwoUpdateGeneral


def compute_l2_loss(batch, labels, gen_pos):
    n_slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
    loss = 0.
    for idx, graph in enumerate(dgl.unbatch(batch)):
        pos = gen_pos[n_slices[idx]:n_slices[idx + 1]]
        gt_pos = labels[n_slices[idx]:n_slices[idx + 1]]
        gen_distmat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)
        gt_distmat = torch.norm(gt_pos.unsqueeze(0) - gt_pos.unsqueeze(1), p=2, dim=-1)
        loss += F.mse_loss(gen_distmat, gt_distmat, reduction='mean')
    return loss, len(batch.batch_num_nodes())


def compute_min_loss(batch, labels, gen_pos, labels_slices, n_gen_samples, return_match_labels=False):
    n_slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
    if labels_slices is None:  # multiple generated mols, one reference mol
        assert n_gen_samples > 1
        cur_idx = 0
        loss = 0.
        for idx, graph in enumerate(dgl.unbatch(batch)):
            num_nodes = batch.batch_num_nodes().tolist()[idx]
            end_idx = cur_idx + num_nodes * n_gen_samples
            all_pos = gen_pos[cur_idx:end_idx].view(n_gen_samples, num_nodes, 3)
            gt_pos = labels[n_slices[idx]:n_slices[idx + 1]]
            min_loss = None
            for n_idx in range(len(all_pos)):
                pos = all_pos[n_idx]
                gen_distmat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)
                gt_distmat = torch.norm(gt_pos.unsqueeze(0) - gt_pos.unsqueeze(1), p=2, dim=-1)
                dist_loss = F.mse_loss(gen_distmat, gt_distmat, reduction='mean')
                if min_loss is None or dist_loss < min_loss:
                    min_loss = dist_loss
            loss += min_loss
    else:
        l_slices = np.cumsum([0] + labels_slices)
        loss = 0.
        match_labels = []
        for idx, graph in enumerate(dgl.unbatch(batch)):
            pos = gen_pos[n_slices[idx]:n_slices[idx + 1]]
            label = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, len(pos), 3)
            min_loss = None
            for l_idx in range(len(label)):
                gt_pos = label[l_idx]
                gen_distmat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)
                gt_distmat = torch.norm(gt_pos.unsqueeze(0) - gt_pos.unsqueeze(1), p=2, dim=-1)
                dist_loss = F.mse_loss(gen_distmat, gt_distmat, reduction='mean')
                if min_loss is None or dist_loss < min_loss:
                    min_loss = dist_loss
                    m_label = label[l_idx]
            match_labels.append(m_label)
            loss += min_loss
    if return_match_labels:
        return loss, len(batch.batch_num_nodes()), torch.cat(match_labels, dim=0)
    else:
        return loss, len(batch.batch_num_nodes())


def compute_wasserstein_loss(batch, labels, gen_pos, labels_slices):
    l_slices = np.cumsum([0] + labels_slices)
    loss = 0.
    for idx, graph in enumerate(dgl.unbatch(batch)):
        num_nodes = graph.number_of_nodes()
        pos = gen_pos[l_slices[idx]:l_slices[idx + 1]].view(-1, num_nodes, 3)  # [n_samples, n_nodes, 3]
        label = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, num_nodes, 3)
        num_samples = pos.shape[0]
        assert pos.shape == label.shape
        gen_distmat = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1)
        gt_distmat = torch.norm(label.unsqueeze(1) - label.unsqueeze(2), p=2, dim=-1)  # [n_samples, n_nodes, n_nodes]

        all_dist_loss = torch.zeros(num_samples, num_samples)
        for i in range(num_samples):
            for j in range(num_samples):
                # todo: F.l1_loss?
                all_dist_loss[i][j] = F.mse_loss(gen_distmat[i], gt_distmat[j],
                                                 reduction='sum')  # [n_samples, n_nodes, n_nodes]

        # enumerate over all permutation (when n_samples is large, may need other bipartite matching algorithm)
        gen_idx, gt_idx = linear_sum_assignment(all_dist_loss.detach().numpy())
        dist_mse = F.mse_loss(gen_distmat[gen_idx, ...], gt_distmat[gt_idx, ...], reduction='none').view(-1, num_nodes * num_nodes)
        wasserstein_loss = torch.mean(dist_mse.mean(-1).sqrt())
        loss += wasserstein_loss
    return loss, len(batch.batch_num_nodes())


def get_init_pos(propose_net_type, batch, labels, noise, gt_aug_ratio=0.05, noise_type='const',
                 n_ref_samples=1, n_gen_samples=1, labels_slices=None, eval_mode=False):
    l_slices = np.cumsum([0] + labels_slices) if labels_slices is not None else None
    if not eval_mode and (propose_net_type != 'gt' and np.random.rand() < gt_aug_ratio):
        # data augmentation step where we feed the correct pos to the model and expect zero delta
        if n_ref_samples == n_gen_samples:
            init_pos = labels
        elif n_ref_samples == 1 and n_gen_samples > 1:
            all_init_pos = []
            n, cur_idx = 0, 0
            for idx, num_nodes in enumerate(batch.batch_num_nodes().tolist()):
                all_init_pos.append(labels[cur_idx:cur_idx + num_nodes])
                n += 1
                if n >= n_gen_samples:
                    cur_idx += num_nodes
                    n = 0
            init_pos = torch.cat(all_init_pos)
        elif n_gen_samples == 1 and n_ref_samples > 1:
            all_init_pos = []
            for idx, num_nodes in enumerate(batch.batch_num_nodes().tolist()):
                label = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, num_nodes, 3)
                rand_idx = random.randint(0, len(label) - 1)
                init_pos = label[rand_idx]
                all_init_pos.append(init_pos)
            init_pos = torch.cat(all_init_pos)
        else:
            raise NotImplementedError(f'n ref samples {n_ref_samples} and n gen samples {n_gen_samples} mismatch')

    elif propose_net_type == 'rdkit' or propose_net_type == 'online_rdkit':
        # initialized the model input with the rdkit pos if the generation is successful
        if n_gen_samples == 1:  # [n_nodes, n_samples, 3]
            init_pos = batch.ndata['rdkit_pos'] + utils.noise_like(batch.ndata['rdkit_pos'], noise_type, noise)

        else:
            # tile batch when n_gen_sample > 1
            # init_pos = batch.ndata['rdkit_pos'].permute(1, 0, 2).reshape(-1, 3)
            # init_pos += utils.noise_like(init_pos, noise_type, noise)
            init_pos = []
            sample_idx = 0
            for idx, graph in enumerate(dgl.unbatch(batch)):
                pos = graph.ndata['rdkit_pos'][:, sample_idx % n_gen_samples] + utils.noise_like(
                    graph.ndata['rdkit_pos'][:, sample_idx % n_gen_samples], noise_type, noise)
                sample_idx += 1
                init_pos.append(pos)
            init_pos = torch.cat(init_pos, dim=0).to(labels)

    elif propose_net_type == 'random':
        # initialized the model input with the random pos
        init_pos = []
        for n_nodes in batch.batch_num_nodes().tolist():
            init_pos.append(torch.randn(n_nodes, 3) * (1 + n_nodes * noise))
        init_pos = torch.cat(init_pos, dim=0).to(labels)

    elif propose_net_type == 'gt':
        # initialized the model with ground truth + noise, but eval with the rdkit init
        if n_ref_samples == n_gen_samples:
            init_pos = labels + utils.noise_like(labels, noise_type, noise, labels_slices)

        elif n_gen_samples == 1 and n_ref_samples > 1:
            all_init_pos = []
            for idx, num_nodes in enumerate(batch.batch_num_nodes().tolist()):
                label = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, num_nodes, 3)
                rand_idx = random.randint(0, len(label) - 1)
                init_pos = label[rand_idx] + utils.noise_like(label[rand_idx], noise_type, noise)
                all_init_pos.append(init_pos)
            init_pos = torch.cat(all_init_pos)
        else:
            raise ValueError('No need to make n_gen_samples > 1 when n_ref_samples = 1 and propose net is gt')
    else:
        raise ValueError(propose_net_type)
    return init_pos


def get_refine_net(refine_net_type, config):
    # baseline
    if refine_net_type == 'equi_se3trans':
        refine_net = EquiSE3Transformer(
            num_layers=config.num_layers,
            atom_feature_size=config.hidden_dim,
            num_channels=config.num_channels,
            num_nlayers=config.num_nlayers,
            num_degrees=config.num_degrees,
            edge_dim=config.hidden_dim,
            div=config.div,
            n_heads=config.n_heads
        )
    elif refine_net_type == 'egnn':
        refine_net = EnEquiNetwork(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            update_x=True,
            act_fn=config.act_fn,
            norm=config.norm
        )
    # our model
    elif refine_net_type == 'ours_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            edge_feat_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            r_feat_mode=config.r_feat_mode,
            energy_h_mode=config.energy_h_mode,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    # our model
    elif refine_net_type == 'ours_o3':
        refine_net = UniTransformerO3TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            edge_feat_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            r_feat_mode=config.r_feat_mode,
            energy_h_mode=config.energy_h_mode,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            num_init_x2h=config.num_init_x2h,
            num_init_h2x=config.num_init_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


class PosNet3D(nn.Module):
    def __init__(self, config, node_type_dim=100, edge_type_dim=len(BOND_TYPES)):
        super(PosNet3D, self).__init__()
        self.refine_net_type = config.model_type
        self.hidden_dim = config.hidden_dim
        self.node_type_dim = node_type_dim
        self.edge_type_dim = edge_type_dim
        self.node_feat_dim = self.node_type_dim + len(NODE_FEATS)
        self.edge_feat_dim = self.edge_type_dim + 1  # edge_conj_feat
        if 'our' in self.refine_net_type and 'mix' in config.energy_h_mode:
            self.node_emb = nn.Linear(self.node_feat_dim, self.hidden_dim * 2, bias=False)
        else:
            self.node_emb = nn.Linear(self.node_feat_dim, self.hidden_dim, bias=False)
        self.edge_emb = nn.Linear(self.edge_feat_dim, self.hidden_dim, bias=False)
        self.refine_net = get_refine_net(self.refine_net_type, config)

    def forward(self, G, init_pos):
        edge_index = torch.stack(G.edges()).to(G.ndata['node_type'].device)
        node_feat = torch.cat([F.one_hot(G.ndata['node_type'], self.node_type_dim).to(G.ndata['node_feat']),
                               G.ndata['node_feat']], dim=-1)
        node_attr = self.node_emb(node_feat)
        edge_feat = torch.cat([F.one_hot(G.edata['edge_type'], self.edge_type_dim).to(G.ndata['node_feat']),
                               G.edata['edge_feat']], dim=-1)
        edge_attr = self.edge_emb(edge_feat)

        # refine coordinates with SE(3)-equivariant network
        G.ndata['x'] = init_pos
        src, dst = edge_index
        G.edata['d'] = init_pos[dst] - init_pos[src]
        G.ndata['f'] = node_attr.unsqueeze(-1)
        G.edata['w'] = edge_attr
        final_pos, all_pos = self.refine_net(G)
        return final_pos, all_pos

    def get_gen_pos(self, propose_net_type, batch, labels, noise, gt_aug_ratio, noise_type='const',
                    n_ref_samples=1, n_gen_samples=1, labels_slices=None, zero_mass_center=False,
                    eval_mode=False, fix_init_pos=None):
        if fix_init_pos is not None:
            init_pos = fix_init_pos
        else:
            if eval_mode:
                assert propose_net_type != 'gt'

            if n_gen_samples == 1:
                init_pos = get_init_pos(propose_net_type, batch, labels, noise, gt_aug_ratio, noise_type,
                                        n_ref_samples, n_gen_samples, labels_slices, eval_mode=eval_mode)
            else:
                tile_batch = []
                for idx, graph in enumerate(dgl.unbatch(batch)):
                    for _ in range(n_gen_samples):
                        tile_batch.append(graph)
                batch = dgl.batch(tile_batch)
                init_pos = get_init_pos(propose_net_type, batch, labels, noise, gt_aug_ratio, noise_type,
                                        n_ref_samples, n_gen_samples, labels_slices, eval_mode=eval_mode)

        if zero_mass_center:
            # make the init_pos zero mass center
            n_slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
            standard_init_pos = []
            for idx, graph in enumerate(dgl.unbatch(batch)):
                pos = init_pos[n_slices[idx]:n_slices[idx + 1]]
                pos = pos - pos.mean(0)
                standard_init_pos.append(pos)
            init_pos = torch.cat(standard_init_pos, dim=0)

        gen_pos, all_pos = self(batch, init_pos)
        return init_pos, gen_pos, all_pos
