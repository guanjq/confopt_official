import os

import torch.nn as nn
import torch.utils.tensorboard
import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from datasets.qm9_property import TARGET_NAMES
from utils import misc as utils_misc
from utils.parsing_args import get_prop_pred_args
from utils.transforms import get_edge_transform

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args, config = get_prop_pred_args()
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
    train_dset = utils_misc.get_prop_dataset(config.data, edge_transform, 'train', args.pre_pos_path, args.pre_pos_filename)
    val_dset = utils_misc.get_prop_dataset(config.data, edge_transform, 'valid', args.pre_pos_path, args.pre_pos_filename)
    test_dset = utils_misc.get_prop_dataset(config.data, edge_transform, 'test', args.pre_pos_path, args.pre_pos_filename)

    logger.info('TrainSet %d | ValSet %d | TestSet %d' % (len(train_dset), len(val_dset), len(test_dset)))

    train_loader = DataLoader(train_dset, batch_size=config.train.batch_size, collate_fn=utils_misc.collate_prop,
                              num_workers=config.train.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, batch_size=config.train.batch_size * 2, collate_fn=utils_misc.collate_prop,
                            num_workers=config.train.num_workers, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size=config.train.batch_size * 2, collate_fn=utils_misc.collate_prop,
                             num_workers=config.train.num_workers, shuffle=False, drop_last=False)

    # Model
    logger.info('Building model...')
    target_index = TARGET_NAMES.index(config.data.target_name)
    target_mean = train_dset.target_mean[target_index]
    target_std = train_dset.target_std[target_index]
    if args.resume is None:
        model = utils_misc.build_prop_pred_model(
            config,
            target_index=target_index,
            target_mean=target_mean,
            target_std=target_std
        )
    else:
        logger.info('Resuming from %s' % args.resume)
        if args.resume_iter is None:
            ckpt_resume = utils_misc.CheckpointManager(
                args.resume, logger=logger, keep_n_ckpt=args.keep_n_ckpt).load_latest()
        else:
            ckpt_resume = utils_misc.CheckpointManager(
                args.resume, logger=logger, keep_n_ckpt=args.keep_n_ckpt).load_with_iteration(args.resume_iter)
        config = ckpt_resume['config']  # override config
        model = utils_misc.build_prop_pred_model(
            config,
            target_index=target_index,
            target_mean=target_mean,
            target_std=target_std
        )
        model.load_state_dict(ckpt_resume['state_dict'])

    model = model.to(args.device)
    logger.info(repr(model))
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay,
                                 amsgrad=True if config.train.opt_type == 'amsgrad' else False)

    if config.train.sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=config.train.sched_factor,
                                                               patience=config.train.sched_patience,
                                                               min_lr=config.train.min_lr
                                                               )
    elif config.train.sched_type == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.num_epochs)
    else:
        raise ValueError(config.train.sched_type)

    if config.train.pred_loss_type == 'mse':
        loss_func = nn.MSELoss()
    else:  # mae
        loss_func = nn.L1Loss()

    if args.resume:
        logger.info('Restoring optimizer and scheduler from %s' % args.resume)
        optimizer.load_state_dict(ckpt_resume['opt_state_dict'])
        scheduler.load_state_dict(ckpt_resume['sche_state_dict'])

    # Main loop
    logger.info('Start training...')

    if args.resume is not None:
        start_it = ckpt_resume['iteration'] + 1
        validate(ckpt_resume['iteration'], val_loader, model, logger, args.device,
                 config.data.target_name, config.train)
    else:
        start_it = 1
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_test_mae = float('inf')
    best_val_epoch = 0
    patience = 0

    torch.cuda.empty_cache()
    for epoch in tqdm.trange(start_it, config.train.num_epochs + 1, dynamic_ncols=True, desc='Training'):
        train(epoch, train_loader, model, loss_func, optimizer, scheduler,
              logger, writer, args.device, config.data.target_name, config.train)
        avg_val_loss, avg_rescale_mae = validate(
            epoch, val_loader, model, logger, args.device, config.data.target_name,
            config.train, scheduler, writer, tag='Val ')
        avg_test_loss, test_rescale_mae = validate(
            epoch, test_loader, model, logger, args.device, config.data.target_name, config.train, tag='Test')

        if avg_val_loss < best_val_loss:
            patience = 0
            best_val_loss = avg_val_loss
            best_val_mae = avg_rescale_mae
            best_test_mae = test_rescale_mae
            best_val_epoch = epoch
            logger.info(
                f'Best val loss achieves: {best_val_loss:.4f} (rescale mae: {best_val_mae:.4f}) at epoch {best_val_epoch} '
                f'(test rescale mae: {best_test_mae: .4f})'
            )
            ckpt_mgr.save(model, optimizer, scheduler, config, avg_val_loss, epoch, logger)
        else:
            patience += 1
            logger.info(
                f'Patience {patience} / {config.train.patience}  '
                f'Best val loss: {best_val_loss:.4f}  (rescale mae: {best_val_mae:.4f}) at epoch {best_val_epoch} '
                f'(test rescale mae: {best_test_mae: .4f})'
            )
        if patience == config.train.patience or epoch == config.train.num_epochs:
            logger.info('Max patience! Stop training and evaluate on the test set!')
            best_ckpt = ckpt_mgr.load_best()
            model.load_state_dict(best_ckpt['state_dict'])
            validate(epoch, test_loader, model, logger, args.device, config.data.target_name, config.train)
            break


def train(epoch, train_loader, model, loss_func, optimizer, scheduler, logger, writer, device,
          target_name, config):
    model.train()
    target_index = TARGET_NAMES.index(target_name)
    it = 0
    num_it = len(train_loader)
    optimizer.zero_grad()
    for batch, labels, meta_info in tqdm.tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}', position=1):
        it += 1
        batch = batch.to(torch.device(device))
        labels = labels.to(device)[:, target_index]
        pred, gen_pos = model(batch, config.pos_type)
        pred_loss = loss_func(pred.view(-1), labels)
        loss = pred_loss / config.n_acc_batch

        loss.backward()
        if it % config.n_acc_batch == 0:
            if config.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if it % config.train_report_iter == 0:
            logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | Lr %.4f * 1e-3' % (
                epoch, it, loss.item(), optimizer.param_groups[0]['lr'] * 1000))

        writer.add_scalar('train/loss', loss, it + epoch * num_it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it + epoch * num_it)
        writer.flush()

    if config.sched_type == 'cos':
        scheduler.step()


def validate(epoch, val_loader, model, logger, device,
             target_name, config, scheduler=None, writer=None, tag='Validate'):
    with torch.no_grad():
        model.eval()
        maes = []
        target_index = TARGET_NAMES.index(target_name)
        for batch, labels, meta_info in tqdm.tqdm(val_loader, dynamic_ncols=True, desc='Validating', leave=None):
            batch = batch.to(torch.device(device))
            labels = labels.to(device)[:, target_index]
            pred, gen_pos = model(batch, config.pos_type)
            mae = (pred.view(-1) - labels).abs()
            maes.append(mae)
        mae = torch.cat(maes, dim=0).cpu()  # [num_examples, num_targets]

    avg_loss = mae.mean()
    mae = 1000 * mae if target_name in ['homo', 'lumo', 'gap', 'zpve', 'u0', 'u298', 'h298', 'g298'] else mae
    logger.info(f'[{tag}] Epoch {epoch:03d} |  Target: {target_name}  Avg loss {avg_loss:.6f}  '
                f'rescale MAE: {mae.mean():.5f} ± {mae.std():.5f}')

    if config.sched_type == 'plateau' and scheduler is not None:
        scheduler.step(avg_loss)

    if writer is not None:
        writer.add_scalar('val/mae', mae.mean(), epoch)
        writer.flush()
    return avg_loss, mae.mean()


if __name__ == '__main__':
    main()
