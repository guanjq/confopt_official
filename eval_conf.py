import os
import argparse
import pickle
from torch.utils.data import DataLoader

from datasets.energy_dgl import ConfDatasetDGL
from utils import misc as utils_misc
from utils.transforms import get_edge_transform
from utils import eval_opt as utils_eval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, default='./data/qm9/qm9_test.pkl')
    parser.add_argument('--data_processed_tag', type=str, default='dgl_processed')
    parser.add_argument('--dset_mode', type=str, default='relax_lowest')
    parser.add_argument('--lowest_thres', type=float, default=0.5)
    parser.add_argument('--rdkit_pos_mode', type=str, default='random')

    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_eval_log', type=eval, default=False, choices=[True, False])

    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--dump_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    utils_misc.seed_all(args.seed)
    if args.save_eval_log:
        logger = utils_misc.get_logger('eval', args.ckpt_path, 'log_eval_lowest.txt')
    else:
        logger = utils_misc.get_logger('eval')
    logger.info(args)

    # Model
    logger.info(f'Loading model from {args.ckpt_path}')
    if args.ckpt_iter is None:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_best()
    else:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_with_iteration(args.ckpt_iter)
    logger.info(f'Loaded model at iteration: {ckpt_restore["iteration"]}  val loss: {ckpt_restore["score"]}')
    ckpt_config = utils_misc.load_config(os.path.join(args.ckpt_path, 'config.yml'))
    logger.info(f'ckpt_config: {ckpt_config}')
    model = utils_misc.build_pos_net(ckpt_config).to(args.device)
    model.load_state_dict(ckpt_restore['state_dict'])
    logger.info(repr(model))
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # Dataset
    edge_transform = get_edge_transform(
        ckpt_config.data.edge_transform_mode, ckpt_config.data.aux_edge_order,
        ckpt_config.data.cutoff, ckpt_config.data.cutoff_pos)
    test_dset = ConfDatasetDGL(args.test_dataset, heavy_only=ckpt_config.data.heavy_only, edge_transform=edge_transform,
                               processed_tag=args.data_processed_tag, rdkit_pos_mode=args.rdkit_pos_mode,
                               mode=args.dset_mode, lowest_thres=args.lowest_thres)
    logger.info('TestSet %d' % (len(test_dset)))
    test_loader = DataLoader(test_dset, batch_size=args.val_batch_size, collate_fn=utils_misc.collate_multi_labels,
                             num_workers=args.num_workers, shuffle=False, drop_last=False)

    # Evaluation
    # utils_eval.validate_rdkit(test_loader, logger, args.device)
    all_gen_results = utils_eval.validate_model(
        ckpt_restore["iteration"], test_loader, model, logger, args.device, prefix='Test', return_all_gen_results=True)
    if args.dump_path:
        with open(args.dump_path, 'wb') as f:
            pickle.dump(all_gen_results, f)


if __name__ == '__main__':
    main()
