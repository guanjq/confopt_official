import dgl
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader

from datasets.energy_dgl import ConfDatasetDGL
from utils import misc as utils_misc
from utils import transforms as utils_trans
from utils.parsing_args import get_conf_opt_args


def get_dataset(config, path, edge_transform_func):
    dset = ConfDatasetDGL(path,
                          edge_transform=edge_transform_func,
                          heavy_only=config.heavy_only,
                          processed_tag=config.processed_tag,
                          rdkit_pos_mode=config.rdkit_pos_mode,
                          mode='lowest',
                          lowest_thres=1.0)
    return dset


def collate(samples):
    graphs, labels, meta_info = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.cat(labels)
    return batched_graph, batched_labels, meta_info


def test_equivariant():
    args, config = get_conf_opt_args()
    config = EasyDict(config)
    utils_misc.seed_all(config.train.seed)
    logger = utils_misc.get_logger('train', None)

    edge_transform = utils_trans.EdgeWithHigherOrder(full=True, max_order=10)
    val_dset = get_dataset(config.data, config.data.val_dataset, edge_transform)

    logger.info('ValSet %d' % (len(val_dset)))
    val_loader = utils_misc.get_data_iterator(
        DataLoader(val_dset, batch_size=1, collate_fn=collate,
                   num_workers=config.train.num_workers, shuffle=False, drop_last=False))

    # Model
    logger.info('Building model...')
    model = utils_misc.build_pos_net(config).to(args.device)
    logger.info(repr(model))
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # Test equivariance
    model.eval()
    batch, labels, meta_info = next(val_loader)
    batch = batch.to(torch.device(args.device))
    labels = labels.to(args.device)

    # original
    init_pos = batch.ndata['rdkit_pos'] + torch.randn_like(batch.ndata['rdkit_pos']) * config.train.noise_std
    _, final_pos, _ = model.get_gen_pos('rdkit', batch, labels, noise=0., gt_aug_ratio=0., fix_init_pos=init_pos)
    # final_pos = model('rdkit', batch, labels, meta_info, fix_init_pos=init_pos)
    print('init pos: ', init_pos)
    print('out pos: ', final_pos)

    loss, n_edges = 0, 0
    slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
    for idx, graph in enumerate(dgl.unbatch(batch)):
        pos = final_pos[slices[idx]:slices[idx + 1]]
        gt_pos = labels[slices[idx]:slices[idx + 1]]
        gen_distmat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)
        gt_distmat = torch.norm(gt_pos.unsqueeze(0) - gt_pos.unsqueeze(1), p=2, dim=-1)
        loss += F.mse_loss(gen_distmat, gt_distmat, reduction='sum')
        n_edges += len(graph.nodes()) ** 2
    print('loss: ', loss / n_edges)

    # translation
    init_pos_trans = init_pos + 1.0
    # final_pos_trans = model(batch, sampled_init_pos=init_pos_trans)
    _, final_pos_trans, _ = model.get_gen_pos('rdkit', batch, labels,
                                              noise=0., gt_aug_ratio=0., fix_init_pos=init_pos_trans)
    print('init pos trans: ', init_pos_trans)
    print('out pos trans: ', final_pos_trans)
    print('out pos trans should be: ', final_pos + 1.0)

    # rotation
    M = np.random.randn(3, 3)
    Q, __ = np.linalg.qr(M)
    Q = torch.from_numpy(Q.astype(np.float32)).to(batch.ndata['x'])

    init_pos_rot = init_pos @ Q
    # final_pos_rot = model(batch, sampled_init_pos=init_pos_rot)
    _, final_pos_rot, _ = model.get_gen_pos('rdkit', batch, labels,
                                            noise=0., gt_aug_ratio=0., fix_init_pos=init_pos_rot)
    print('init pos rot: ', init_pos_rot)
    print('out pos rot: ', final_pos_rot)
    print('out pos rot should be: ', final_pos @ Q)

    # rotation + trans
    init_pos_mix = init_pos @ Q + 1.0
    # final_pos_mix = model(batch, sampled_init_pos=init_pos_mix)
    _, final_pos_mix, _ = model.get_gen_pos('rdkit', batch, labels,
                                            noise=0., gt_aug_ratio=0., fix_init_pos=init_pos_mix)
    print('init pos mix: ', init_pos_mix)
    print('out pos mix: ', final_pos_mix)
    print('out pos mix should be: ', final_pos @ Q + 1.0)


if __name__ == '__main__':
    # run it like:
    # python test_equivariance.py --config configs/qm9_default.yml --model_type ours_o2 --energy_h_mode share_cos_sim
    test_equivariant()
