import os

import torch.utils.tensorboard
import tqdm
import argparse
from utils import eval_opt as utils_eval
from utils import misc as utils_misc
from utils.transforms import get_edge_transform
from utils.eval_opt import generate_multi_confs
from utils.evaluation import evaluate_conf
import pickle
import copy
from functools import partial
import multiprocessing
import numpy as np
from rdkit import Chem

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_iter', type=int, default=None)
    # parser.add_argument('--dump_path', type=str, default=None)
    parser.add_argument('--eval_propose_net_type', type=str, choices=['online_rdkit', 'random'])
    parser.add_argument('--eval_noise', type=float)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    utils_misc.seed_all(args.seed)
    logger = utils_misc.get_logger('eval_sampling', None)
    logger.info(args)
    ckpt_config = utils_misc.load_config(os.path.join(args.ckpt_path, 'config.yml'))
    ckpt_config.eval.eval_propose_net_type = args.eval_propose_net_type
    ckpt_config.eval.eval_noise = args.eval_noise

    # Dataset and dataloader
    edge_transform = get_edge_transform(
        ckpt_config.data.edge_transform_mode, ckpt_config.data.aux_edge_order,
        ckpt_config.data.cutoff, ckpt_config.data.cutoff_pos)
    test_dset = utils_misc.get_conf_dataset(ckpt_config.data, ckpt_config.data.test_dataset, edge_transform,
                                            rdkit_pos_mode='online', rdkit_mol=False, n_gen_samples='auto',
                                            mode='relax_lowest')
    logger.info('TestSet %d' % len(test_dset))

    # Model
    logger.info(f'Loading model from {args.ckpt_path}')
    if args.ckpt_iter is None:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_best()
    else:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_with_iteration(args.ckpt_iter)
    model = utils_misc.build_pos_net(ckpt_config).to(args.device)
    model.load_state_dict(ckpt_restore['state_dict'])
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # utils_eval.validate_sampling_rdkit(test_dset, ckpt_config, args.device, logger)
    # test(ckpt_restore["iteration"], test_dset, model, logger, args.device, ckpt_config, mode='rdkit', save_dir=None, cal_scores=True)
    test(ckpt_restore["iteration"], test_dset, model, logger, args.device, ckpt_config, save_dir=None, cal_scores=True)


def test(it, test_dset, model, logger, device, config, save_dir, mode='model', cal_scores=False, size_limit=None):
    ref_mols, gen_mols, all_gen_results = generate_multi_confs(
        dset=test_dset,
        model=model,
        eval_propose_net_type=config.eval.eval_propose_net_type,
        val_batch_size=config.train.batch_size * 2,
        eval_noise=config.eval.eval_noise,
        device=device,
        heavy_only=config.data.heavy_only,
        ff_opt=config.eval.ff_opt,
        n_samples='auto', mode=mode, return_gen_results=True, size_limit=size_limit)

    if save_dir:
        if not os.path.exists(os.path.join(save_dir, 'test')):
            os.mkdir(os.path.join(save_dir, 'test'))
        out_path = os.path.join(save_dir, 'test', 'step%d.pkl' % it)
        with open(out_path, 'wb') as fout:
            pickle.dump(all_gen_results, fout)
        logger.info('Save generated samples to %s done!' % out_path)

    if cal_scores:
        data_list = []
        for r in all_gen_results:
            rdmol = copy.deepcopy(r['mol'])
            rdmol.RemoveAllConformers()
            if config.data.heavy_only:
                rdmol = Chem.RemoveHs(rdmol)
            pos_ref = torch.from_numpy(r['gt_pos'])
            if mode == 'rdkit':
                pos_gen = torch.from_numpy(r['rdkit_pos'])
            else:
                pos_gen = torch.from_numpy(r['gen_pos'])
            data_list.append((rdmol, pos_ref, pos_gen))

        func = partial(evaluate_conf, useFF=False, threshold=config.eval.delta)

        covs = []
        mats = []
        junks = []
        with multiprocessing.Pool(16) as pool:
            for result in pool.starmap(func, tqdm.tqdm(data_list, total=len(data_list))):
                covs.append(result[0])
                mats.append(result[1])
                junks.append(result[2])
        covs = np.array(covs)
        mats = np.array(mats)
        junks = np.array(junks)

        logger.info(
            'Coverage Mean: %.4f | Coverage Median: %.4f | Mismatch Mean: %.4f | Mismatch Median: %.4f | '
            'Match Mean: %.4f | Match Median: %.4f' % (
                covs.mean(), np.median(covs), junks.mean(), np.median(junks), mats.mean(), np.median(mats)))


if __name__ == '__main__':
    main()
