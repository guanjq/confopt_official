import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from models.utils import GaussianSmearing, MLP


class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10. ** 2
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.r_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, G, h, x):
        """Forward pass of the linear layer
        Args:
            h: dict of node-features
            x: coordinates
            G: minibatch of (homo)graphs
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        edge_index = G.edges()
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = G.edata['w']  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None
        rel_x = x[dst] - x[src]
        r = torch.sum(rel_x ** 2, -1, keepdim=True)

        if self.num_r_gaussian > 1:
            r_feat = self.r_expansion(r)
        else:
            r_feat = r

        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        if edge_feat is None:
            mij = self.edge_mlp(torch.cat([r_feat, hi, hj], -1))
        else:
            mij = self.edge_mlp(torch.cat([edge_feat, r_feat, hi, hj], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            delta_x = scatter_sum((xi - xj) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x

        return h, x


class EnEquiNetwork(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian,
                 update_x=True, act_fn='relu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, G, return_h=False):
        x = G.ndata['x']
        h = G.ndata['f'].squeeze(-1)

        all_x = [x]
        all_h = [h]
        for l_idx, layer in enumerate(self.net):
            h, x = layer(G, h, x)
            all_x.append(x)
            all_h.append(h)

        if return_h:
            return x, all_x, all_h
        else:
            return x, all_x
