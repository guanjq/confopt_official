import numpy as np
import torch
import torch.nn as nn

from models.baseline_models.equivariant_attention.fibers import Fiber
from models.baseline_models.equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res


class TFN(nn.Module):
    """SE(3) equivariant GCN"""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 3)
        self.Gblock, self.FCblock = blocks

    def _build_gcn(self, fibers, out_dim):

        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            Gblock.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            Gblock.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G.edata['d'], self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        h = h['0'].squeeze(-1)
        for layer in self.FCblock:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, n_heads: int = 1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees * self.num_channels)}

        blocks = self._build_gcn(self.fibers, 3)
        self.Gblock, self.FCblock = blocks

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G.edata['d'], self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        h = h['0'].squeeze(-1)
        for layer in self.FCblock:
            h = layer(h)

        return h


class EquiSE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, n_heads: int = 1):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(2, num_degrees * self.num_channels)}

        self.Gblock = self._build_gcn(self.fibers)
        m_in = self.num_degrees * self.num_channels
        self.out_w = nn.Parameter(torch.randn(m_in, 1) / np.sqrt(m_in), requires_grad=True)

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        return nn.ModuleList(Gblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G.edata['d'], self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for idx, layer in enumerate(self.Gblock):
            h = layer(h, G=G, r=r, basis=basis)

        final_pos = (self.out_w.unsqueeze(0) * h['1']).sum(1)
        all_pos = [final_pos]
        return final_pos, all_pos
