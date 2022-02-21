import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from datasets.qm9_property import TARGET_NAMES
from utils import misc as utils_misc
from utils.transforms import get_edge_transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/qm9_property')
    parser.add_argument('--split_file', type=str, default='./data/qm9_property/split.npz')
    parser.add_argument('--data_processed_tag', type=str, default='dgl_processed')
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    # Eval
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ckpt_path', type=str, default='./logs_prop_pred')
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_eval_log', type=eval, default=False, choices=[True, False])

    parser.add_argument('--pre_pos_path', type=str, default=None)
    parser.add_argument('--pre_pos_filename', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    utils_misc.seed_all(args.seed)
    if args.save_eval_log:
        logger = utils_misc.get_logger('eval', args.ckpt_path, 'log_eval.txt')
    else:
        logger = utils_misc.get_logger('eval')
    logger.info(args)
    logger.info(f'Loading model from {args.ckpt_path}')
    if args.ckpt_iter is None:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_best()
    else:
        ckpt_restore = utils_misc.CheckpointManager(args.ckpt_path, logger=logger).load_with_iteration(args.ckpt_iter)
    logger.info(f'Loaded model at iteration: {ckpt_restore["iteration"]}  val loss: {ckpt_restore["score"]}')
    ckpt_config = utils_misc.load_config(os.path.join(args.ckpt_path, 'config.yml'))
    logger.info(f'ckpt_config: {ckpt_config}')

    edge_transform = get_edge_transform(
        ckpt_config.data.edge_transform_mode, ckpt_config.data.aux_edge_order,
        ckpt_config.data.cutoff, ckpt_config.data.cutoff_pos)
    target_name = ckpt_config.data.target_name
    target_index = TARGET_NAMES.index(target_name)
    # override data path
    ckpt_config.data.dataset_path = args.dataset_path
    ckpt_config.data.split_file = args.split_file

    test_dset = utils_misc.get_prop_dataset(
        ckpt_config.data, edge_transform, 'test', args.pre_pos_path, args.pre_pos_filename)
    logger.info('TestSet %d' % (len(test_dset)))

    test_loader = DataLoader(test_dset, batch_size=args.val_batch_size, collate_fn=utils_misc.collate_prop,
                             num_workers=args.num_workers, shuffle=False, drop_last=False)

    ckpt_state = ckpt_restore['state_dict']
    model = utils_misc.build_prop_pred_model(
        ckpt_config, target_index=target_index,
        target_mean=ckpt_state['target_mean'] if 'target_mean' in ckpt_state else None,
        target_std=ckpt_state['target_std'] if 'target_std' in ckpt_state else None
    ).to(args.device)
    model.load_state_dict(ckpt_restore['state_dict'])
    logger.info(repr(model))
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    with torch.no_grad():
        model.eval()
        maes = []
        for batch, labels, meta_info in tqdm.tqdm(test_loader, dynamic_ncols=True, desc='Testing', leave=None):
            batch = batch.to(torch.device(args.device))
            labels = labels.to(args.device)[:, target_index]
            pred, gen_pos = model(batch, ckpt_config.train.pos_type)
            mae = (pred.view(-1) - labels).abs()
            maes.append(mae)
        mae = torch.cat(maes, dim=0).cpu()  # [num_examples]

    avg_loss = mae.mean()
    mae = 1000 * mae if target_name in ['homo', 'lumo', 'gap', 'zpve', 'u0', 'u298', 'h298', 'g298'] else mae
    logger.info(f'[Test] Epoch {ckpt_restore["iteration"]:03d} |  Target: {target_name}  Avg loss {avg_loss:.6f}  '
                f'rescale MAE: {mae.mean():.5f} ± {mae.std():.5f}')


if __name__ == '__main__':
    main()
