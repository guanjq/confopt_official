import argparse
import copy
import os
import pickle

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.energy_dgl import ConfDatasetDGL
from utils import misc as utils_misc
from utils.chem import get_conformer_energies, get_molecule_force_field
from utils.conf import add_conformer
from utils.eval_opt import calculate_rmsd
from utils.transforms import get_edge_transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='qm9')
    parser.add_argument('--heavy_only', type=eval, default=True, choices=[True, False])
    parser.add_argument('--dset_mode', type=str, default='relax_lowest', choices=['lowest', 'relax_lowest'])
    parser.add_argument('--lowest_thres', type=float, default=0.5)
    parser.add_argument('--data_processed_tag', type=str, default='dgl_processed')
    parser.add_argument('--test_dataset', type=str, default='./data/qm9/qm9_test.pkl')
    parser.add_argument('--rdkit_pos_mode', type=str, default='random')
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_gen_results', type=eval, default=False, choices=[True, False])
    parser.add_argument('--tag', type=str, default='test')

    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path_list', type=str, nargs='+', default=None)

    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--best_k', type=int, default=5)
    parser.add_argument('--ff_opt', type=eval, default=False, choices=[True, False])
    parser.add_argument('--ff_rmsd_tol', type=float, default=0.1)
    parser.add_argument('--filter_pos', type=eval, default=True)

    args = parser.parse_args()
    return args


def filter_conf(dset, gen_pos_list, best_k, ff_opt, rmsd_tol=0.1, logger=None):
    avg_energy, best_energy = [], []
    avg_rmsd, best_rmsd = [], []
    all_best_pos = []
    n_fail = 0
    n_ff_min = 0
    for idx, (data, label, meta_info) in enumerate(tqdm(dset, desc='Filtering')):
        multi_gen_pos = gen_pos_list[idx]
        label = label.cpu().numpy().astype(np.float64)
        # get conf energy
        rdmol = copy.deepcopy(meta_info['rdmol'])
        rdmol.RemoveAllConformers()
        rdmol = add_conformer(rdmol, multi_gen_pos)
        try:
            # if ff minimize, set an accepted rmsd threshold
            if ff_opt:
                for i in range(len(multi_gen_pos)):
                    ff = get_molecule_force_field(rdmol, conf_id=i)
                    ff.Minimize()
                    new_gen_pos = rdmol.GetConformer(i).GetPositions()
                    _, _, _, diff_rmsd = calculate_rmsd(rdmol, multi_gen_pos[i], new_gen_pos)
                    if diff_rmsd < rmsd_tol:
                        n_ff_min += 1
                        multi_gen_pos[i] = new_gen_pos
            energies = get_conformer_energies(rdmol)
        except:
            n_fail += 1
            all_best_pos.append(multi_gen_pos[:best_k])
            continue

        sort = np.argsort(energies)
        keep = sort[:best_k]
        best_pos = []
        for i in keep:
            best_pos.append(multi_gen_pos[i])

        all_rmsd = []
        for gen_pos in best_pos:
            _, _, rms, heavy_rms = calculate_rmsd(rdmol, label, gen_pos)
            all_rmsd.append(heavy_rms)

        avg_energy.append(np.mean(energies[keep]))
        best_energy.append(energies[keep[0]])
        avg_rmsd.append(np.mean(all_rmsd))
        best_rmsd.append(np.min(all_rmsd))
        all_best_pos.append(np.array(best_pos))

    logger.info(f'best_k: {best_k}  ff_opt: {ff_opt}  rmsd_tol: {rmsd_tol}')
    logger.info('avg energy: {:.6f}  best energy: {:.6f}  avg rmsd: {:.6f}  best rmsd: {:.6f} n_fail: {:d} n_ff_min: {:d}'.format(
        np.mean(avg_energy), np.mean(best_energy), np.mean(avg_rmsd), np.mean(best_rmsd), n_fail, n_ff_min))
    return all_best_pos


def eval_gen_rmsd(dset, gen_pos_list, logger):
    dset_all_rmsd = []
    avg_rmsd, best_rmsd = [], []
    for idx, (data, label, meta_info) in enumerate(tqdm(dset, desc='Evaluating')):
        multi_gen_pos = gen_pos_list[idx]
        label = label.cpu().numpy().astype(np.float64)
        # get conf energy
        rdmol = copy.deepcopy(meta_info['rdmol'])
        rdmol.RemoveAllConformers()
        rdmol = add_conformer(rdmol, multi_gen_pos)
        all_rmsd = []
        for gen_pos in multi_gen_pos:
            _, _, rms, heavy_rms = calculate_rmsd(rdmol, label, gen_pos)
            all_rmsd.append(heavy_rms)
        dset_all_rmsd.append(all_rmsd)
        avg_rmsd.append(np.mean(all_rmsd))
        best_rmsd.append(np.min(all_rmsd))
    logger.info('avg rmsd: {:.6f}  best rmsd: {:.6f}'.format(np.mean(avg_rmsd), np.mean(best_rmsd)))
    return np.array(dset_all_rmsd)  # [num_examples, 10]


def sample_rmsd(rmsd_list, repeat, logger):
    n_samples = len(rmsd_list[0])
    n_examples = len(rmsd_list)
    all_mean_rmsd, all_median_rmsd = [], []
    for i in range(repeat):
        sample_id = np.random.choice(np.arange(n_samples), size=n_examples)
        sampled = np.array([data[sample_id[idx]] for idx, data in enumerate(rmsd_list)])
        all_mean_rmsd.append(np.mean(sampled))
        all_median_rmsd.append(np.median(sampled))
    logger.info(f'sampled {repeat}: mean RMSD: {np.mean(all_mean_rmsd):.4f} +/- {np.std(all_mean_rmsd):.4f} '
                f'median RMSD: {np.mean(all_median_rmsd):.4f} +/- {np.std(all_median_rmsd):.4f}')


def main():
    args = get_args()
    utils_misc.seed_all(args.seed)
    log_dir = utils_misc.get_new_log_dir(root=args.dump_dir, prefix=args.prefix, tag=args.tag)
    logger = utils_misc.get_logger('eval', log_dir, 'log_dump.txt')
    logger.info(args)

    test_dset = ConfDatasetDGL(args.test_dataset, heavy_only=args.heavy_only, edge_transform=None,
                               processed_tag=args.data_processed_tag, rdkit_pos_mode=args.rdkit_pos_mode,
                               mode=args.dset_mode, lowest_thres=args.lowest_thres)
    logger.info('TestSet %d' % (len(test_dset)))

    # Dump confs generated by RDKit and Models
    all_rdkit_pos = []
    for (data, label, meta_info) in tqdm(test_dset):
        all_rdkit_pos.append(meta_info['all_rdkit_pos'])

    logger.info('Eval RDKit generate pos: ')
    if args.filter_pos:
        all_rdkit_pos = filter_conf(
            test_dset, all_rdkit_pos, args.best_k, ff_opt=False, rmsd_tol=args.ff_rmsd_tol, logger=logger)
    rdkit_all_rmsd = eval_gen_rmsd(test_dset, all_rdkit_pos, logger)
    sample_rmsd(rdkit_all_rmsd, 10, logger)
    np.save('rdkit_all_rmsd.npy', rdkit_all_rmsd)
    with open(os.path.join(log_dir, f'rdkit_gen_conf.pkl'), 'wb') as f:
        pickle.dump(all_rdkit_pos, f)

    for ckpt_path in args.ckpt_path_list:
        # Model
        logger.info(f'Loading model from {ckpt_path}')
        ckpt_restore = utils_misc.CheckpointManager(ckpt_path, logger=logger).load_best()
        logger.info(f'Loaded model at iteration: {ckpt_restore["iteration"]}  val loss: {ckpt_restore["score"]}')
        ckpt_config = utils_misc.load_config(os.path.join(ckpt_path, 'config.yml'))
        logger.info(f'ckpt_config: {ckpt_config}')
        model = utils_misc.build_pos_net(ckpt_config).to(args.device)
        model.load_state_dict(ckpt_restore['state_dict'])
        # logger.info(repr(model))
        logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')
        edge_transform = get_edge_transform(
            ckpt_config.data.edge_transform_mode, ckpt_config.data.aux_edge_order)
        test_dset = ConfDatasetDGL(args.test_dataset, heavy_only=ckpt_config.data.heavy_only,
                                   edge_transform=edge_transform, processed_tag=args.data_processed_tag,
                                   rdkit_pos_mode=args.rdkit_pos_mode,
                                   mode=args.dset_mode, lowest_thres=args.lowest_thres)
        test_loader = DataLoader(test_dset, batch_size=args.val_batch_size,
                                 collate_fn=utils_misc.collate_multi_labels,
                                 num_workers=args.num_workers, shuffle=False, drop_last=False)

        n = 0
        all_gen_pos = []
        for batch, _, batch_meta, _ in tqdm(test_loader, dynamic_ncols=True, desc='Validating', leave=None):
            batch = batch.to(torch.device(args.device))
            multi_init_pos = all_rdkit_pos[n:n + len(batch_meta)]
            n += len(batch_meta)
            tmp_gen_pos = [[] for _ in range(len(batch_meta))]
            for loop_i in range(len(all_rdkit_pos[0])):
                with torch.no_grad():
                    init_pos = [pos[loop_i] for pos in multi_init_pos]
                    init_pos = np.concatenate(init_pos, axis=0)
                    init_pos = torch.from_numpy(init_pos).to(torch.float32).to(args.device)
                    gen_pos, _ = model(batch, init_pos)  # gen_pos: [N, 3]
                    gen_pos = gen_pos.cpu().numpy().astype(np.float64)

                slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
                for idx, graph in enumerate(dgl.unbatch(batch)):
                    pos = gen_pos[slices[idx]:slices[idx + 1]]
                    tmp_gen_pos[idx].append(pos)
            all_gen_pos += tmp_gen_pos

        logger.info(f'Eval model generated pos: (from {ckpt_path})')
        if args.filter_pos:
            all_gen_pos = filter_conf(
                test_dset, all_gen_pos, args.best_k, ff_opt=args.ff_opt, rmsd_tol=args.ff_rmsd_tol, logger=logger)
        gen_all_rmsd = eval_gen_rmsd(test_dset, all_gen_pos, logger)
        sample_rmsd(gen_all_rmsd, 10, logger)
        if 'our' in model.refine_net_type:
            save_prefix = model.refine_net_type + '_' + model.refine_net.energy_h_mode
        else:
            save_prefix = model.refine_net_type
        np.save(f'{save_prefix}_gen_all_rmsd.npy', gen_all_rmsd)
        with open(os.path.join(log_dir, f'{save_prefix}_gen_conf.pkl'), 'wb') as f:
            pickle.dump(all_gen_pos, f)


if __name__ == '__main__':
    main()
