import os

import torch.utils.tensorboard
from tqdm.auto import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from models.conf_model import compute_min_loss, get_init_pos
from utils import eval_opt as utils_eval
from utils import misc as utils_misc
from utils.parsing_args import get_conf_opt_args
from utils.transforms import get_edge_transform
from utils.evaluation import evaluate_conf
import copy
import numpy as np
import multiprocessing
from functools import partial
from time import time

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args, config = get_conf_opt_args()
    # Logging
    logger = utils_misc.get_logger('eval', None)
    config = EasyDict(config)
    utils_misc.seed_all(config.train.seed)
    logger.info(args)
    edge_transform = get_edge_transform(
        config.data.edge_transform_mode, config.data.aux_edge_order, config.data.cutoff, config.data.cutoff_pos)

    test_dset = utils_misc.get_conf_dataset(config.data, config.data.test_dataset, edge_transform,
                                            rdkit_mol=False, n_gen_samples='auto')
    logger.info('TestSet %d' % (len(test_dset)))

    data_list = []
    for G, labels, meta_info in tqdm(test_dset):
        rdmol = copy.deepcopy(meta_info['ori_rdmol'])
        rdmol.RemoveAllConformers()
        pos_ref = labels
        pos_gen = G.ndata['rdkit_pos'].permute(1, 0, 2)
        data_list.append((rdmol, pos_ref, pos_gen))

    func = partial(evaluate_conf, useFF=True, threshold=config.eval.delta)

    covs = []
    mats = []
    with multiprocessing.Pool(8) as pool:
        for result in pool.starmap(func, tqdm(data_list, total=len(data_list))):
            covs.append(result[0])
            mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
          (covs.mean(), np.median(covs), mats.mean(), np.median(mats)))


if __name__ == '__main__':
    main()
