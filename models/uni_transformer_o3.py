import dgl
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum

from models import utils
from models.utils import GaussianSmearing, MLP, get_r_feat


class BaseX2HO3AttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', energy_h_mode='basic', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.energy_h_mode = energy_h_mode
        self.out_fc = out_fc

        kv_input_dim = input_dim * 3 + edge_feat_dim * 3 + r_feat_dim * 3
        q_input_dim = input_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(kv_input_dim, output_dim * 2, hidden_dim, norm=norm, act_fn=act_fn)
        self.v_jk_split = output_dim
        self.hq_func = MLP(q_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(self.v_jk_split, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w, id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj):
        N = h.size(0)
        hi, hj, hk = h[id3_i], h[id3_j], h[id3_k]
        ef_ji, ef_ki, ef_kj = edge_feat[edgeid_ji], edge_feat[edgeid_ki], edge_feat[edgeid_kj]
        r_ji, r_ki, r_kj = r_feat[edgeid_ji], r_feat[edgeid_ki], r_feat[edgeid_kj]

        # multi-head attention
        kv_input_ijk = torch.cat([ef_ji, r_ji,
                                  ef_ki, r_ki,
                                  ef_kj, r_kj,
                                  hi, hj, hk], dim=-1)
        kv_input_ikj = torch.cat([ef_ki, r_ki,
                                  ef_ji, r_ji,
                                  ef_kj, r_kj,
                                  hi, hk, hj], dim=-1)

        k_ijk = self.hk_func(kv_input_ijk).view(-1, self.n_heads, self.output_dim // self.n_heads)
        k_ikj = self.hk_func(kv_input_ikj).view(-1, self.n_heads, self.output_dim // self.n_heads)

        vjk_ijk = self.hv_func(kv_input_ijk)
        vj_ijk = vjk_ijk[..., :self.v_jk_split]
        vk_ijk = vjk_ijk[..., self.v_jk_split:]
        vjk_ikj = self.hv_func(kv_input_ikj)
        vj_ikj = vjk_ikj[..., :self.v_jk_split]
        vk_ikj = vjk_ikj[..., self.v_jk_split:]

        if self.ew_net_type == 'r':
            e_w_ji = self.ew_net(r_ji)
            e_w_ki = self.ew_net(r_ki)
        elif self.ew_net_type == 'm':
            e_w_ji = (self.ew_net(vj_ijk) + self.ew_net(vj_ikj)) / 2
            e_w_ki = (self.ew_net(vk_ijk) + self.ew_net(vk_ikj)) / 2
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
            e_w_ji = e_w[edgeid_ji]
            e_w_ki = e_w[edgeid_ki]
        else:
            e_w_ji = 1.
            e_w_ki = 1.

        vj_ijk = vj_ijk * e_w_ji
        vk_ijk = vk_ijk * e_w_ki
        vj_ikj = vj_ikj * e_w_ji
        vk_ikj = vk_ikj * e_w_ki

        if self.energy_h_mode == 'basic':
            v_ijk = vj_ijk + vk_ijk
            v_ikj = vj_ikj + vk_ikj
            v = (v_ijk + v_ikj) / 2
            v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)
        else:
            vj_ijk = vj_ijk * (hi - hj)
            vk_ijk = vk_ijk * (hi - hk)
            vj_ikj = vj_ikj * (hi - hj)
            vk_ikj = vk_ikj * (hi - hk)

            v_ijk = vj_ijk + vk_ijk
            v_ikj = vj_ikj + vk_ikj
            v = (v_ijk + v_ikj) / 2
            v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)[id3_i]

        # Compute attention weights
        e_ijk = (k_ijk * q).sum(-1)
        e_ijk = e_ijk / np.sqrt(k_ijk.shape[-1])
        a_ijk = scatter_softmax(e_ijk, id3_i, dim=0)
        e_ikj = (k_ikj * q).sum(-1)
        e_ikj = e_ikj / np.sqrt(k_ikj.shape[-1])
        a_ikj = scatter_softmax(e_ikj, id3_i, dim=0)
        alpha = (a_ijk + a_ikj) / 2

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, id3_i, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XO3AttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', energy_h_mode='basic'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type
        self.energy_h_mode = energy_h_mode

        kv_input_dim = input_dim * 3 + edge_feat_dim * 3 + r_feat_dim * 3
        q_input_dim = input_dim
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads * 2, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(q_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w, id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji,
                edgeid_kj):
        N = h.size(0)
        hi, hj, hk = h[id3_i], h[id3_j], h[id3_k]
        ef_ji, ef_ki, ef_kj = edge_feat[edgeid_ji], edge_feat[edgeid_ki], edge_feat[edgeid_kj]
        r_ji, r_ki, r_kj = r_feat[edgeid_ji], r_feat[edgeid_ki], r_feat[edgeid_kj]

        # multi-head attention
        kv_input_ijk = torch.cat([ef_ji, r_ji,
                                  ef_ki, r_ki,
                                  ef_kj, r_kj,
                                  hi, hj, hk], dim=-1)
        kv_input_ikj = torch.cat([ef_ki, r_ki,
                                  ef_ji, r_ji,
                                  ef_kj, r_kj,
                                  hi, hk, hj], dim=-1)

        k_ijk = self.xk_func(kv_input_ijk).view(-1, self.n_heads, self.output_dim // self.n_heads)
        k_ikj = self.xk_func(kv_input_ikj).view(-1, self.n_heads, self.output_dim // self.n_heads)

        if self.ew_net_type == 'r':
            e_w_ji = self.ew_net(r_ji)
            e_w_ki = self.ew_net(r_ki)
        elif self.ew_net_type == 'm':
            e_w_ji = 1.
            e_w_ki = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
            e_w_ji = e_w[edgeid_ji]
            e_w_ki = e_w[edgeid_ki]
        else:
            e_w_ji = 1.
            e_w_ki = 1.

        rel_x = rel_x.unsqueeze(1)

        vjk_ijk = self.xv_func(kv_input_ijk)
        vj_ijk = vjk_ijk[..., :self.n_heads] * e_w_ji
        vk_ijk = vjk_ijk[..., self.n_heads:] * e_w_ki
        vj_ijk = vj_ijk.unsqueeze(-1) * rel_x[edgeid_ji]
        vk_ijk = vk_ijk.unsqueeze(-1) * rel_x[edgeid_ki]
        v_ijk = vj_ijk + vk_ijk
        vjk_ikj = self.xv_func(kv_input_ikj)
        vj_ikj = vjk_ikj[..., :self.n_heads] * e_w_ji
        vk_ikj = vjk_ikj[..., self.n_heads:] * e_w_ki
        vj_ikj = vj_ikj.unsqueeze(-1) * rel_x[edgeid_ji]
        vk_ikj = vk_ikj.unsqueeze(-1) * rel_x[edgeid_ki]
        v_ikj = vj_ikj + vk_ikj
        v = (v_ijk + v_ikj) / 2
        v = v.view(-1, self.n_heads, 3)

        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)[id3_i]

        # Compute attention weights
        e_ijk = (k_ijk * q).sum(-1)
        e_ijk = e_ijk / np.sqrt(k_ijk.shape[-1])
        a_ijk = scatter_softmax(e_ijk, id3_i, dim=0)
        e_ikj = (k_ikj * q).sum(-1)
        e_ikj = e_ikj / np.sqrt(k_ikj.shape[-1])
        a_ikj = scatter_softmax(e_ikj, id3_i, dim=0)
        alpha = (a_ijk + a_ikj) / 2

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, id3_i, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO3TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 r_feat_mode='basic', energy_h_mode='basic', ew_net_type='r',
                 x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r2_min = r_min
        self.r2_max = r_max
        self.num_node_types = num_node_types
        self.r_feat_mode = r_feat_mode  # ['origin', 'basic', 'sparse']
        self.energy_h_mode = energy_h_mode
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        if r_feat_mode == 'origin':
            self.r_expansion = None
            r_feat_dim = 1
        elif r_feat_mode == 'basic':
            self.r_expansion = GaussianSmearing(self.r2_min, self.r2_max, num_gaussians=num_r_gaussian)
            r_feat_dim = num_r_gaussian
        else:
            self.r_expansion = GaussianSmearing(self.r2_min, self.r2_max, num_gaussians=4)
            r_feat_dim = num_node_types * num_node_types * 4

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HO3AttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim, r_feat_dim,
                                  act_fn=act_fn, norm=norm,
                                  ew_net_type=self.ew_net_type, energy_h_mode=self.energy_h_mode,
                                  out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XO3AttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim, r_feat_dim,
                                  act_fn=act_fn, norm=norm,
                                  ew_net_type=self.ew_net_type, energy_h_mode=self.energy_h_mode)
            )

    def forward(self, G, h, x, id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj, e_w=None):
        edge_index = G.edges()
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = G.edata['w']  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None
        node_type = G.ndata['remap_node_type']
        rel_x = x[dst] - x[src]
        r = torch.sqrt(torch.sum(rel_x ** 2, -1, keepdim=True) + utils.VERY_SMALL_NUMBER)

        h_in = h
        for i in range(self.num_x2h):
            r_feat = get_r_feat(r, self.r_expansion, node_type, edge_index, self.r_feat_mode)
            h_out = self.x2h_layers[i](h_in, r_feat, edge_feat, edge_index, e_w, id3_i, id3_j, id3_k, edgeid_ki,
                                       edgeid_ji, edgeid_kj)
            h_in = h_out
        x2h_out = h_in

        # print('x2h out', x2h_out.sum())
        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            r_feat = get_r_feat(r, self.r_expansion, node_type, edge_index, self.r_feat_mode)
            delta_x = self.h2x_layers[i](new_h, rel_x, r_feat, edge_feat, edge_index, e_w, id3_i, id3_j, id3_k,
                                         edgeid_ki, edgeid_ji, edgeid_kj)
            x = x + delta_x
            rel_x = x[dst] - x[src]
            r = torch.sqrt(torch.sum(rel_x ** 2, -1, keepdim=True) + utils.VERY_SMALL_NUMBER)

        return x2h_out, x


class UniTransformerO3TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r', r_feat_mode='basic', energy_h_mode='basic',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.ew_net_type = ew_net_type  # [r, m, none]
        self.r_feat_mode = r_feat_mode  # [basic, sparse]
        self.energy_h_mode = energy_h_mode

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        if self.ew_net_type == 'global':
            if r_feat_mode == 'origin':
                self.r_expansion = None
                r_feat_dim = 1
            elif r_feat_mode == 'basic':
                self.r_expansion = GaussianSmearing(num_gaussians=num_r_gaussian)
                r_feat_dim = num_r_gaussian
            else:
                self.r_expansion = GaussianSmearing(num_gaussians=4)
                r_feat_dim = num_node_types * num_node_types * 4
            self.edge_pred_layer = MLP(r_feat_dim, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO3(num_blocks={self.num_blocks}, num_layers={self.num_layers}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'r_feat_mode={self.r_feat_mode}, energy_h_mode={self.energy_h_mode}）\n' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO3TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            r_feat_mode=self.r_feat_mode, energy_h_mode=self.energy_h_mode, ew_net_type=self.ew_net_type,
            x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO3TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                r_feat_mode=self.r_feat_mode, energy_h_mode=self.energy_h_mode, ew_net_type=self.ew_net_type,
                x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def forward(self, G, return_h=False):
        x = G.ndata['x']
        h = G.ndata['f'].squeeze(-1)
        edge_index = G.edges()
        full_src, full_dst = edge_index
        id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj = get_triplet(G)
        h, _ = self.init_h_emb_layer(G, h, x, id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj)

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            r = torch.sqrt(torch.sum((x[full_dst] - x[full_src]) ** 2, -1, keepdim=True) + utils.VERY_SMALL_NUMBER)
            if self.ew_net_type == 'global':
                r_feat = get_r_feat(r, self.r_expansion, G.ndata['remap_node_type'], edge_index, self.r_feat_mode)
                logits = self.edge_pred_layer(r_feat)
                edge_prob = torch.sigmoid(logits)
            else:
                edge_prob = torch.ones_like(r)

            if self.cutoff_mode == 'radius':
                e_w = torch.where(r < self.r_max, edge_prob, torch.zeros_like(r))
                valid_ew_index = torch.nonzero(e_w, as_tuple=True)[0]
                sub_G = dgl.edge_subgraph(G, valid_ew_index, preserve_nodes=True)
                e_w = e_w[valid_ew_index]
            else:
                sub_G = G
                e_w = edge_prob

            id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj = get_triplet(sub_G)

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(sub_G, h, x, id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj, e_w)
            all_x.append(x)
            all_h.append(h)

        if return_h:
            return x, all_x, all_h
        else:
            return x, all_x


def get_triplet(G):
    with torch.no_grad():
        N = len(G.nodes())
        device = G.ndata['node_type'].device
        src, dst = torch.stack(G.edges()).to(device)
        adj = torch.zeros([N, N], dtype=torch.long, device=device)
        adj[src, dst] = 1

        # (k, j)->i  2-tuple
        rep_idx = adj[src].nonzero(as_tuple=True)  # replicated arange with #neighbors
        id3_i = src[rep_idx[0]]
        id3_j = dst[rep_idx[0]]
        id3_k = rep_idx[1]

        sel = (id3_j <= id3_k).nonzero(as_tuple=True)
        id3_j, id3_i, id3_k = id3_j[sel], id3_i[sel], id3_k[sel]

        atomsid_to_edgeid = -torch.ones([N, N], dtype=torch.long)
        atomsid_to_edgeid[src, dst] = torch.arange(len(src))
        atomsid_to_edgeid = atomsid_to_edgeid.to(device)
        edgeid_ki = atomsid_to_edgeid[id3_k, id3_i]
        edgeid_ji = atomsid_to_edgeid[id3_j, id3_i]
        # edgeid_kj = atomsid_to_edgeid[id3_k, id3_j]

        r_bonds = G.edata['rotatable_bond']
        identity_sel = (id3_j == id3_k).to(dtype=torch.float)
        r_bond_sel = (r_bonds[edgeid_ki] + r_bonds[edgeid_ji] > 0).to(dtype=torch.float)
        # r_bond_sel = (r_bonds[edgeid_ki] + r_bonds[edgeid_ji] + r_bonds[edgeid_kj] > 0).to(dtype=torch.float)
        sel = (identity_sel + r_bond_sel).nonzero(as_tuple=True)
        id3_j, id3_i, id3_k = id3_j[sel], id3_i[sel], id3_k[sel]

        atomsid_to_edgeid = -torch.ones([N, N], dtype=torch.long)
        atomsid_to_edgeid[src, dst] = torch.arange(len(src))
        atomsid_to_edgeid = atomsid_to_edgeid.to(device)
        edgeid_ki = atomsid_to_edgeid[id3_k, id3_i]
        edgeid_ji = atomsid_to_edgeid[id3_j, id3_i]
        edgeid_kj = atomsid_to_edgeid[id3_k, id3_j]

        return id3_i, id3_j, id3_k, edgeid_ki, edgeid_ji, edgeid_kj
