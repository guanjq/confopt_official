import argparse
import pickle

import numpy as np
from tqdm.auto import tqdm

from datasets.energy_dgl import ConfDatasetDGL
from utils import misc as utils_misc
from utils.eval_opt import calculate_rmsd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heavy_only', type=eval, default=True, choices=[True, False])
    parser.add_argument('--dset_mode', type=str, default='relax_lowest', choices=['lowest', 'relax_lowest'])
    parser.add_argument('--lowest_thres', type=float, default=0.5)
    parser.add_argument('--rdkit_pos_mode', type=str, default='random')
    parser.add_argument('--test_dataset', type=str, default='./data/qm9/qm9_test.pkl')
    parser.add_argument('--data_processed_tag', type=str, default='dgl_processed')

    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--pkl_path', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    utils_misc.seed_all(args.seed)
    logger = utils_misc.get_logger('eval')
    logger.info(args)

    test_dset = ConfDatasetDGL(args.test_dataset, heavy_only=args.heavy_only, edge_transform=None,
                               processed_tag=args.data_processed_tag, rdkit_pos_mode=args.rdkit_pos_mode,
                               mode=args.dset_mode, lowest_thres=args.lowest_thres)
    logger.info('TestSet %d' % (len(test_dset)))

    with open(args.pkl_path, 'rb') as f:
        dump_pos = pickle.load(f)

    rms_list, heavy_rms_list = [], []
    for idx, (data, labels, meta) in enumerate(tqdm(test_dset)):
        gt_pos = labels.numpy().astype(np.float64)
        all_rms, all_heavy_rms = [], []
        for conf_id in range(len(dump_pos[idx])):
            pos = dump_pos[idx][conf_id]
            _, _, rms, heavy_rms = calculate_rmsd(meta['rdmol'], gt_pos, pos)
            all_rms.append(rms)
            if heavy_rms is not None:
                all_heavy_rms.append(heavy_rms)

        rms_list.append(np.mean(all_rms))
        if len(all_heavy_rms) > 0:
            heavy_rms_list.append(np.mean(all_heavy_rms))
    print(f'mean RMSD: {np.mean(rms_list):.6f}  median RMSD: {np.median(rms_list):.6f}')
    print(f'heavy mean RMSD: {np.mean(heavy_rms_list):.6f}  heavy median RMSD: {np.median(heavy_rms_list):.6f}')


if __name__ == '__main__':
    main()
