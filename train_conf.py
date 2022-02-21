import os

import torch.utils.tensorboard
import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from models.conf_model import compute_min_loss, get_init_pos
from utils import eval_opt as utils_eval
from utils import misc as utils_misc
from utils.parsing_args import get_conf_opt_args
from utils.transforms import get_edge_transform

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
    edge_transform = get_edge_transform(
        config.data.edge_transform_mode, config.data.aux_edge_order, config.data.cutoff, config.data.cutoff_pos)
    train_dset = utils_misc.get_conf_dataset(config.data, config.data.train_dataset, edge_transform,
                                             mode=config.data.dset_mode)
    val_dset = utils_misc.get_conf_dataset(config.data, config.data.val_dataset, edge_transform,
                                           mode=config.data.dset_mode)
    test_dset = utils_misc.get_conf_dataset(config.data, config.data.test_dataset, edge_transform,
                                            mode=config.data.dset_mode)
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

    logger.info(repr(model))
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
            utils_eval.validate_model(ckpt_resume['iteration'], val_loader, model, logger, args.device, writer=writer)
        else:
            start_it = 1
        best_val_loss = float('inf')
        best_val_iter = 0
        patience = 0
        logger.info('Evaluate RDKit baseline on the validation set')
        utils_eval.validate_rdkit(val_loader, logger, args.device, writer=writer)

        torch.cuda.empty_cache()
        for it in tqdm.trange(start_it, config.train.max_iters + 1, dynamic_ncols=True, desc='Training'):
            train(it, train_iterator, model, optimizer, logger, writer, args.device, config.train)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = utils_eval.validate_model(it, val_loader, model, logger, args.device,
                                                         writer=writer, prefix='Validate')
                if scheduler:
                    scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = avg_val_loss
                    best_val_iter = it
                    logger.info(f'Best val loss achieves: {best_val_loss:.4f} at iter {best_val_iter}')
                    ckpt_mgr.save(model, optimizer, scheduler, config, avg_val_loss, it, logger)
                else:
                    patience += 1
                    logger.info(f'Patience {patience} / {config.train.patience}  '
                                f'Best val loss: {best_val_loss:.4f} at iter {best_val_iter}')
                if patience == config.train.patience:
                    logger.info('Max patience! Stop training and evaluate on the test set!')
                    best_ckpt = ckpt_mgr.load_best()
                    model.load_state_dict(best_ckpt['state_dict'])
                    utils_eval.validate_model(it, test_loader, model, logger, args.device, prefix='Test')
                    logger.info('Evaluate RDKit baseline on the test set')
                    utils_eval.validate_rdkit(test_loader, logger, args.device)
                    break

    except KeyboardInterrupt:
        logger.info('Terminating...')


def train(it, train_iterator, model, optimizer, logger, writer, device, config):
    model.train()
    optimizer.zero_grad()
    for _ in range(config.n_acc_batch):
        batch, labels, meta_info, labels_slices = next(train_iterator)
        batch = batch.to(torch.device(device))
        labels = labels.to(device)

        with torch.no_grad():
            init_pos = get_init_pos(config.propose_net_type, batch, labels,
                                    noise=config.noise_std, gt_aug_ratio=config.gt_aug_ratio,
                                    noise_type=config.noise_type,
                                    n_ref_samples=10, n_gen_samples=1, labels_slices=labels_slices)

        gen_pos, all_pos = model(batch, init_pos)
        conf_loss, n, match_labels = compute_min_loss(
            batch, labels, gen_pos, labels_slices, n_gen_samples=1, return_match_labels=True)
        loss = conf_loss / n

        loss = loss / config.n_acc_batch
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
    optimizer.step()

    if it % config.train_report_iter == 0:
        logger.info('[Train] Iter %04d | Loss %.6f | Lr %.4f ' % (it, loss.item(), optimizer.param_groups[0]['lr']))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.flush()


if __name__ == '__main__':
    main()
