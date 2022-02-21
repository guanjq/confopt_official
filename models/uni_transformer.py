import dgl
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum

from models import utils
from models.utils import GaussianSmearing, MLP, get_h_dist, get_r_feat


class BaseX2HAttLayer(nn.Module):
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

        # attention key func
        if self.energy_h_mode in ['mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim']:
            kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim + 1
        else:
            kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        if self.energy_h_mode in ['mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim']:
            self.in_hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
            self.eq_hv_func = MLP(kv_input_dim, n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        else:
            self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        if 'mix' in self.energy_h_mode:
            self.hq_func = MLP(input_dim * 2, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        else:
            self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        if 'mix' in self.energy_h_mode:
            in_h = h[..., :self.input_dim]
            hi, hj = in_h[dst], in_h[src]
            eq_h = h[..., self.input_dim:]
            eq_hi, eq_hj = eq_h[dst], eq_h[src]
        else:
            hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        if self.energy_h_mode == 'mix_distance':
            h_dist = get_h_dist('euclidean', eq_hi, eq_hj)
        elif self.energy_h_mode == 'mix_cos_sim':
            h_dist, eq_hj_norm = get_h_dist('cos_sim', eq_hi, eq_hj)
        elif self.energy_h_mode == 'share_distance':
            h_dist = get_h_dist('euclidean', hi, hj)
        elif self.energy_h_mode == 'share_cos_sim':
            h_dist, hj_norm = get_h_dist('cos_sim', hi, hj)
        else:
            h_dist = None
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)
        if h_dist is not None:
            kv_input = torch.cat([kv_input, h_dist], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        if 'mix' in self.energy_h_mode or 'share' in self.energy_h_mode:
            in_v = self.in_hv_func(kv_input)
            eq_v = self.eq_hv_func(kv_input)
            v = torch.cat([in_v, eq_v], -1)
        else:
            v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        if self.energy_h_mode in ['mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim']:
            in_v = v[..., :self.output_dim]
            eq_v = v[..., self.output_dim:]
            in_v = in_v.view(-1, self.n_heads, self.output_dim // self.n_heads)  # [E, heads, H/heads]
            if self.energy_h_mode == 'mix_distance':
                eq_v = eq_v.unsqueeze(-1) * (eq_hi - eq_hj).unsqueeze(1)  # [E, heads, H]
            elif self.energy_h_mode == 'mix_cos_sim':
                eq_v = eq_v.unsqueeze(-1) * (eq_hj / eq_hj_norm).unsqueeze(1)
            elif self.energy_h_mode == 'share_distance':
                eq_v = eq_v.unsqueeze(-1) * (hi - hj).unsqueeze(1)
            elif self.energy_h_mode == 'share_cos_sim':
                eq_v = eq_v.unsqueeze(-1) * (hj / hj_norm).unsqueeze(1)
        else:
            v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        if self.energy_h_mode in ['mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim']:
            in_m = alpha.unsqueeze(-1) * in_v  # (E, heads, H_per_head)
            in_output = scatter_sum(in_m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
            in_output = in_output.view(-1, self.output_dim)
            if self.out_fc:
                if 'mix' in self.energy_h_mode:
                    in_output = self.node_output(torch.cat([in_output, in_h], -1))
                else:
                    in_output = self.node_output(torch.cat([in_output, h], -1))
            eq_m = alpha.unsqueeze(-1) * eq_v  # (E, heads, H_per_head)
            eq_output = scatter_sum(eq_m, dst, dim=0, dim_size=N)  # (N, heads, H)
            eq_output = eq_output.mean(1)
            if 'mix' in self.energy_h_mode:
                output = torch.cat([in_output, eq_output], -1)
            else:
                output = in_output + eq_output
        else:
            m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
            output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
            output = output.view(-1, self.output_dim)
            if self.out_fc:
                output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
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

        if self.energy_h_mode in ['mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim']:
            kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim + 1
        else:
            kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        if 'mix' in self.energy_h_mode:
            self.xq_func = MLP(input_dim * 2, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        else:
            self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        if 'mix' in self.energy_h_mode:
            in_h = h[..., :self.input_dim]
            hi, hj = in_h[dst], in_h[src]
            eq_h = h[..., self.input_dim:]
            eq_hi, eq_hj = eq_h[dst], eq_h[src]
        else:
            hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        if self.energy_h_mode == 'mix_distance':
            h_dist = get_h_dist('euclidean', eq_hi, eq_hj)
        elif self.energy_h_mode == 'mix_cos_sim':
            h_dist, eq_hj_norm = get_h_dist('cos_sim', eq_hi, eq_hj)
        elif self.energy_h_mode == 'share_distance':
            h_dist = get_h_dist('euclidean', hi, hj)
        elif self.energy_h_mode == 'share_cos_sim':
            h_dist, hj_norm = get_h_dist('cos_sim', hi, hj)
        else:
            h_dist = None
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)
        if h_dist is not None:
            kv_input = torch.cat([kv_input, h_dist], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)   # [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
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
        self.r2_min = r_min ** 2 if r_min >= 0 else -(r_min ** 2)
        self.r2_max = r_max ** 2
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
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim, r_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, energy_h_mode=self.energy_h_mode, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim, r_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, energy_h_mode=self.energy_h_mode)
            )

    def forward(self, G, h, x, e_w=None):
        edge_index = G.edges()
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = G.edata['w']  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None
        node_type = G.ndata['remap_node_type']
        rel_x = x[dst] - x[src]
        r = torch.sum(rel_x ** 2, -1, keepdim=True)

        h_in = h
        for i in range(self.num_x2h):
            r_feat = get_r_feat(r, self.r_expansion, node_type, edge_index, self.r_feat_mode)
            h_out = self.x2h_layers[i](h_in, r_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            r_feat = get_r_feat(r, self.r_expansion, node_type, edge_index, self.r_feat_mode)
            delta_x = self.h2x_layers[i](new_h, rel_x, r_feat, edge_feat, edge_index, e_w=e_w)
            x = x + delta_x
            rel_x = x[dst] - x[src]
            r = torch.sum(rel_x ** 2, -1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
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
                self.r_expansion = GaussianSmearing(0., r_max ** 2, num_gaussians=num_r_gaussian)
                r_feat_dim = num_r_gaussian
            else:
                self.r_expansion = GaussianSmearing(0., r_max ** 2, num_gaussians=4)
                r_feat_dim = num_node_types * num_node_types * 4
            self.edge_pred_layer = MLP(r_feat_dim, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'r_feat_mode={self.r_feat_mode}, energy_h_mode={self.energy_h_mode}）\n' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
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
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
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
        h, _ = self.init_h_emb_layer(G, h, x)

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

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(sub_G, h, x, e_w)
            all_x.append(x)
            all_h.append(h)

        if return_h:
            return x, all_x, all_h
        else:
            return x, all_x
