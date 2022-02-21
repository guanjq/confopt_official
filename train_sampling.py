import os

import torch.utils.tensorboard
import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from models.conf_model import compute_min_loss, compute_wasserstein_loss, get_init_pos
from utils import eval_opt as utils_eval
from utils import misc as utils_misc
from utils.parsing_args import get_conf_opt_args
from utils.transforms import get_edge_transform
from utils.eval_opt import generate_multi_confs
from utils.evaluation import evaluate_conf
import pickle
import copy
from functools import partial
import multiprocessing
import numpy as np
import dgl
from rdkit import Chem

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args, config = get_conf_opt_args()
    # Logging
    if args.logging:
        log_dir = utils_misc.get_new_log_dir(root=args.log_dir, prefix=config['data']['dataset_name'], tag=args.tag)
        logger = utils_misc.get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = utils_misc.CheckpointManager(log_dir, logger=logger, keep_n_ckpt=args.keep_n_ckpt)
        # save config
        utils_misc.save_config(os.path.join(log_dir, 'config.yml'), config)
    else:
        logger = utils_misc.get_logger('train', None)
        writer = utils_misc.BlackHole()
        ckpt_mgr = utils_misc.BlackHole()

    config = EasyDict(config)
    utils_misc.seed_all(config.train.seed)
    logger.info(args)

    # Dataset and dataloader
    edge_transform = get_edge_transform(
        config.data.edge_transform_mode, config.data.aux_edge_order, config.data.cutoff, config.data.cutoff_pos)

    train_dset = utils_misc.get_conf_dataset(config.data, config.data.train_dataset, edge_transform,
                                             rdkit_mol=False,
                                             rdkit_pos_mode=config.data.rdkit_pos_mode,
                                             n_ref_samples=config.train.n_ref_samples,
                                             n_gen_samples=config.train.n_gen_samples,
                                             mode=config.data.dset_mode)
    val_dset = utils_misc.get_conf_dataset(config.data, config.data.val_dataset, edge_transform,
                                           rdkit_mol=False,
                                           # rdkit_pos_mode=config.data.rdkit_pos_mode,
                                           rdkit_pos_mode='online',
                                           n_ref_samples=config.train.n_ref_samples,
                                           n_gen_samples=config.train.n_gen_samples,
                                           mode=config.data.dset_mode)
    test_dset = utils_misc.get_conf_dataset(config.data, config.data.test_dataset, edge_transform,
                                            rdkit_pos_mode='online',
                                            rdkit_mol=False, n_gen_samples='auto', mode='relax_lowest')
    logger.info('TrainSet %d | ValSet %d | TestSet %d' % (len(train_dset), len(val_dset), len(test_dset)))

    train_iterator = utils_misc.get_data_iterator(
        DataLoader(
            train_dset, batch_size=config.train.batch_size, collate_fn=utils_misc.collate_multi_labels,
            num_workers=config.train.num_workers, prefetch_factor=8, shuffle=True, drop_last=True
        ))
    val_loader = DataLoader(
        val_dset, batch_size=config.train.batch_size * 2, collate_fn=utils_misc.collate_multi_labels,
        num_workers=config.train.num_workers, prefetch_factor=8, shuffle=False, drop_last=False,
        )
    test_loader = DataLoader(
        test_dset, batch_size=config.train.batch_size * 2, collate_fn=utils_misc.collate_multi_labels,
        num_workers=config.train.num_workers, prefetch_factor=8, shuffle=False, drop_last=False,
    )

    # Model
    logger.info('Building model...')
    if args.resume is None:
        model = utils_misc.build_pos_net(config).to(args.device)

    else:
        logger.info('Resuming from %s' % args.resume)
        ckpt_mgr_resume = utils_misc.CheckpointManager(args.resume, logger=logger, keep_n_ckpt=args.keep_n_ckpt)
        if args.resume_iter is None:
            ckpt_resume = ckpt_mgr_resume.load_latest()
        else:
            ckpt_resume = ckpt_mgr_resume.load_with_iteration(args.resume_iter)
        ckpt_config = ckpt_resume['config']
        model = utils_misc.build_pos_net(ckpt_config).to(args.device)
        model.load_state_dict(ckpt_resume['state_dict'])
        config.update(ckpt_config)

    # logger.info(repr(model))
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train.lr,
                                 weight_decay=config.train.weight_decay,
                                 betas=(config.train.beta1, config.train.beta2)
                                 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=config.train.sched_factor,
                                                           patience=config.train.sched_patience,
                                                           min_lr=config.train.min_lr
                                                           )

    if args.resume:
        logger.info('Restoring optimizer and scheduler from %s' % args.resume)
        optimizer.load_state_dict(ckpt_resume['opt_state_dict'])
        scheduler.load_state_dict(ckpt_resume['sche_state_dict'])

    # Main loop
    logger.info('Start training...')
    try:
        if args.resume is not None:
            start_it = ckpt_resume['iteration'] + 1
            utils_eval.validate_sampling_model(ckpt_resume['iteration'],
                                               val_dset, val_loader, model, config, args.device, logger)
        else:
            start_it = 1
        best_val_loss = float('inf')
        best_val_iter = 0
        patience = 0
        # logger.info('Evaluate RDKit baseline on the validation set')
        # utils_eval.validate_sampling_rdkit(val_dset, config, args.device, logger)
        # if args.pretrained_model_path is not None:
        #     logger.info('Pretrain model performance on the validation set')
        #     utils_eval.validate_model(0, val_loader, model, logger, args.device, prefix='Validate')

        torch.cuda.empty_cache()
        # test(0, val_dset, model, logger, args.device, config, save_dir=None, mode='rdkit', cal_scores=True, size_limit=200)
        # test(0, test_dset, model, logger, args.device, config, save_dir=None, mode='rdkit', cal_scores=True)
        for it in tqdm.trange(start_it, config.train.max_iters + 1, dynamic_ncols=True, desc='Training'):
            train(it, train_iterator, model, optimizer, logger, writer, args.device, config.train)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it, val_loader, model, logger, writer, args.device, config)
                # test(it, val_dset, model, logger, args.device, config, save_dir=None, cal_scores=True, size_limit=200)

                if scheduler:
                    scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = avg_val_loss
                    best_val_iter = it
                    logger.info(f'Best val loss achieves: {best_val_loss:.4f} at iter {best_val_iter}')
                    ckpt_mgr.save(model, optimizer, scheduler, config, avg_val_loss, it, logger)
                    test(it, test_dset, model, logger, args.device, config, log_dir, cal_scores=True)
                else:
                    patience += 1
                    logger.info(f'Patience {patience} / {config.train.patience}  '
                                f'Best val loss: {best_val_loss:.4f} at iter {best_val_iter}')
                if patience == config.train.patience:
                    logger.info('Max patience! Stop training and evaluate on the test set!')
                    best_ckpt = ckpt_mgr.load_best()
                    model.load_state_dict(best_ckpt['state_dict'])
                    test(it, test_dset, model, logger, args.device, config, log_dir, cal_scores=True)
                    # utils_eval.validate_sampling_model(it, test_dset, test_loader, model, config, args.device, logger, prefix='Test', quick_mode=False)
                    # logger.info('Evaluate RDKit baseline on the test set')
                    # utils_eval.validate_sampling_rdkit(test_dset, config, args.device, logger, quick_mode=False)
                    break

    except KeyboardInterrupt:
        logger.info('Terminating...')


# Train and validation
def train(it, train_iterator, model, optimizer, logger, writer, device, config):
    model.train()
    optimizer.zero_grad()
    for _ in range(config.n_acc_batch):
        batch, labels, meta_info, labels_slices = next(train_iterator)
        batch = batch.to(torch.device(device))
        labels = labels.to(device)

        # t1 = time.time()
        if config.loss_type == 'wasserstein':
            tile_batch, tile_labels = [], []
            for idx, graph in enumerate(dgl.unbatch(batch)):
                for _ in range(config.n_gen_samples):
                    tile_batch.append(graph)
            tile_batch = dgl.batch(tile_batch)
        else:
            tile_batch = batch

        with torch.no_grad():
            init_pos = get_init_pos(config.propose_net_type, tile_batch, labels,
                                    noise=config.noise_std, gt_aug_ratio=config.gt_aug_ratio,
                                    noise_type=config.noise_type,
                                    n_ref_samples=config.n_ref_samples,
                                    n_gen_samples=config.n_gen_samples,
                                    labels_slices=labels_slices)

        # print('init pos shape: ', init_pos.shape, labels.shape, labels_slices, tile_batch.number_of_nodes())
        gen_pos, all_pos = model(tile_batch, init_pos)
        # t2 = time.time()
        if config.loss_type == 'min':
            loss, n, match_labels = compute_min_loss(
                batch, labels, gen_pos, labels_slices, n_gen_samples=1, return_match_labels=True)
        elif config.loss_type == 'wasserstein':
            loss, n = compute_wasserstein_loss(batch, labels, gen_pos, labels_slices)
        else:
            raise NotImplementedError
        loss = loss / n
        loss = loss / config.n_acc_batch
        loss.backward()

        # t3 = time.time()
        # print(f'forward time: {t2 - t1} backward time: {t3 - t2}')

    ori_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
    optimizer.step()

    if it % config.train_report_iter == 0:
        logger.info('[Train] Iter %04d | Loss %.6f | Lr %.4f | Grad Norm %.4f ' % (
            it, loss.item(), optimizer.param_groups[0]['lr'], ori_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', ori_grad_norm, it)
    writer.flush()


def validate(it, val_loader, model, logger, writer, device, config):
    model.eval()
    sum_loss, sum_n = 0, 0
    for batch, labels, batch_meta, labels_slices in tqdm.tqdm(
            val_loader, dynamic_ncols=True, desc='Validating', leave=None):
        batch = batch.to(torch.device(device))
        labels = labels.to(device)

        if config.train.loss_type == 'wasserstein':
            tile_batch = []
            for idx, graph in enumerate(dgl.unbatch(batch)):
                for _ in range(config.train.n_gen_samples):
                    tile_batch.append(graph)
            tile_batch = dgl.batch(tile_batch)
        else:
            tile_batch = batch

        with torch.no_grad():
            init_pos = get_init_pos(config.eval.eval_propose_net_type, tile_batch, labels,
                                    noise=config.eval.eval_noise, gt_aug_ratio=0.,
                                    n_ref_samples=config.train.n_ref_samples,
                                    n_gen_samples=config.train.n_gen_samples,
                                    labels_slices=labels_slices, eval_mode=True)
            gen_pos, all_pos = model(tile_batch, init_pos)

        if config.train.loss_type == 'min':
            batch_loss, batch_n, match_labels = compute_min_loss(
                batch, labels, gen_pos, labels_slices, n_gen_samples=1, return_match_labels=True)
        elif config.train.loss_type == 'wasserstein':
            batch_loss, batch_n = compute_wasserstein_loss(batch, labels, gen_pos, labels_slices)
        else:
            raise NotImplementedError
        sum_loss += batch_loss
        sum_n += batch_n

    loss = sum_loss / sum_n
    logger.info('[Val] Iter %04d | Loss %.6f ' % (it, loss.item()))
    writer.add_scalar('val/loss', loss, it)
    writer.flush()
    return loss


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
        with multiprocessing.Pool(16) as pool:
            for result in pool.starmap(func, tqdm.tqdm(data_list, total=len(data_list))):
                covs.append(result[0])
                mats.append(result[1])
        covs = np.array(covs)
        mats = np.array(mats)

        logger.info('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
                    (covs.mean(), np.median(covs), mats.mean(), np.median(mats)))


if __name__ == '__main__':
    main()
