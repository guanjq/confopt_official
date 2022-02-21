import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce

from utils.chem import BOND_TYPES


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    adj_mats = [torch.eye(adj.size(0)).long(), binarize(adj + torch.eye(adj.size(0)).long())]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    # print(adj_mats)

    order_mat = torch.zeros_like(adj)
    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return order_mat


class EdgeWithInCutoff(object):
    def __init__(self, cutoff, pos_key='pos', max_order=10, num_types=len(BOND_TYPES)):
        self.cutoff = cutoff
        self.pos_key = pos_key
        self.max_order = max_order
        self.num_types = num_types

    def __call__(self, data):
        N = data['num_atoms']
        adj = to_dense_adj(data['edge_index'], max_num_nodes=N).squeeze(0)  # special case, such as 'N.N#C/C=C\CN=C=O'
        adj_order = get_higher_order_adj_matrix(adj, self.max_order)  # (N, N)
        pos = data[self.pos_key] if len(data[self.pos_key].shape) == 2 else data[self.pos_key][0]
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        aux_edge_index = radius_graph(pos, r=self.cutoff)
        aux_edge_order = adj_order[aux_edge_index[0], aux_edge_index[1]].to(torch.long)
        aux_edge_type = aux_edge_order + self.num_types - 1
        return aux_edge_index, aux_edge_type, aux_edge_order


class EdgeWithHigherOrder(object):
    def __init__(self, full, max_order=10, num_types=len(BOND_TYPES)):
        self.full = full
        self.max_order = max_order
        self.num_types = num_types

    def __call__(self, data):
        N, edge_index, edge_type = data['num_atoms'], data['edge_index'], data['edge_type']
        adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0)  # special case, such as 'N.N#C/C=C\CN=C=O'
        type_mat = to_dense_adj(edge_index, edge_attr=edge_type, max_num_nodes=N).squeeze(0)  # (N, N)
        adj_order = get_higher_order_adj_matrix(adj, self.max_order)  # (N, N)

        if self.full:
            adj_order = torch.where(adj_order == 0,
                                    (torch.ones_like(adj_order) - torch.eye(N).long()) * self.max_order,
                                    adj_order)

        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder
        aux_edge_index, aux_edge_type = dense_to_sparse(type_new)
        _, aux_edge_order = dense_to_sparse(adj_order)

        aux_edge_index, aux_edge_type = coalesce(aux_edge_index, aux_edge_type.long(), N, N)
        _, aux_edge_order = coalesce(aux_edge_index, aux_edge_order.long(), N, N)
        return aux_edge_index, aux_edge_type, aux_edge_order


def get_edge_transform(edge_transform_mode, aux_edge_order=10, cutoff=10., cutoff_pos='pos'):
    if edge_transform_mode == 'aux_edge':
        edge_transform = EdgeWithHigherOrder(full=False, max_order=aux_edge_order)
    elif edge_transform_mode == 'full_edge':
        edge_transform = EdgeWithHigherOrder(full=True, max_order=aux_edge_order)
    elif edge_transform_mode == 'cutoff':
        edge_transform = EdgeWithInCutoff(cutoff=cutoff, pos_key=cutoff_pos)
    else:
        edge_transform = None
    return edge_transform


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()
