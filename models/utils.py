import numpy as np
import torch
import torch.nn as nn

from utils.transforms import outer_product

VERY_SMALL_NUMBER = 1e-16


def noise_like(tensor, noise_type, noise, label_slices=None):
    if noise_type == 'expand':
        noise_tensor = randn_like_expand(tensor, label_slices, sigma1=noise)
    elif noise_type == 'const':
        noise_tensor = randn_like_with_clamp(tensor) * noise
    else:
        raise NotImplementedError
    return noise_tensor


def randn_like_with_clamp(tensor, clamp_std=3):
    noise_tensor = torch.randn_like(tensor)
    return torch.clamp(noise_tensor, min=-clamp_std, max=clamp_std)


def randn_like_expand(tensor, label_slices, sigma0=0.01, sigma1=10, num_sigma=50):
    # max_noise_std = 20 if noise >= 0.5 else 5
    # noise_std_list = np.linspace(0, 1, 11)[:-1].tolist() + np.linspace(1, max_noise_std, 20).tolist()
    # idx = np.random.randint(0, len(noise_std_list))
    # noise_tensor = torch.randn_like(tensor) * noise_std_list[idx]
    sigmas = np.exp(np.linspace(np.log(sigma0), np.log(sigma1), num_sigma))
    batch_noise_std = np.random.choice(sigmas, len(label_slices))
    batch_noise_std = torch.tensor(batch_noise_std, dtype=torch.float32)
    batch_noise_std = torch.repeat_interleave(batch_noise_std, torch.tensor(label_slices))
    # print('noise tensor shape: ', tensor.shape, 'noise std shape: ', batch_noise_std.shape)
    # print('label slices: ', label_slices)
    # print('batch noise std: ', batch_noise_std)
    noise_tensor = torch.randn_like(tensor) * batch_noise_std.unsqueeze(-1).to(tensor)
    # print('noise tensor: ', noise_tensor.shape)
    return noise_tensor


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def convert_dgl_to_batch(dgl_batch, device):
    batch_idx = []
    for idx, num_nodes in enumerate(dgl_batch.batch_num_nodes().tolist()):
        batch_idx.append(torch.ones(num_nodes, dtype=torch.long) * idx)
    batch_idx = torch.cat(batch_idx).to(device)
    return batch_idx


def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def get_r_feat(r, r_exp_func, node_type, edge_index, mode):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat