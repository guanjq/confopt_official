import collections
import logging
import os
import random
import time

import dgl
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from easydict import EasyDict

from models.conf_model import PosNet3D
from models.prop_model import PropPredNet
from utils.chem import BOND_TYPES
from datasets.energy_dgl import ConfDatasetDGL
from datasets.qm9_property import QM9PropertyDataset


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def save_config(path, config):
    with open(path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name, log_dir=None, log_filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    stream_handler = TqdmLoggingHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def get_data_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_prop(samples):
    graphs, labels, meta_info = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_labels, meta_info


def batchify_labels(labels):
    if len(labels[0].shape) == 2:
        # [num_nodes, 3]
        batched_labels = torch.cat(labels)
        labels_slices = None
    else:
        # [num_ref_confs, num_nodes, 3]
        batched_labels = []
        labels_slices = []
        for i in range(len(labels)):
            labels_view = labels[i].view(-1, 3)
            labels_slices.append(len(labels_view))
            batched_labels.append(labels_view)
        batched_labels = torch.cat(batched_labels)
    return batched_labels, labels_slices


def collate_multi_labels(samples):
    graphs, labels, meta_info = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels, labels_slices = batchify_labels(labels)
    return batched_graph, batched_labels, meta_info, labels_slices


def build_pos_net(config):
    model = PosNet3D(
        config.model,
        node_type_dim=100,
        edge_type_dim=len(BOND_TYPES) + config.data.aux_edge_order
    )
    return model


def build_prop_pred_model(config, target_index, target_mean, target_std):
    model = PropPredNet(
        config.model,
        aux_edge_order=config.data.aux_edge_order,
        target_idx=target_index,
        target_mean=target_mean,
        target_std=target_std
    )
    return model


def get_conf_dataset(config, path, edge_transform_func, **kwargs):
    dset = ConfDatasetDGL(
        path,
        edge_transform=edge_transform_func,
        heavy_only=config.heavy_only,
        processed_tag=config.processed_tag,
        lowest_thres=config.lowest_thres,
        **kwargs
    )
    return dset


def get_prop_dataset(config, edge_transform_func, split_type, pre_pos_path=None, pre_pos_filename=None):
    if pre_pos_path is not None and pre_pos_filename is not None:
        model_pos_path = os.path.join(pre_pos_path, split_type, pre_pos_filename)
    else:
        model_pos_path = None
    dset = QM9PropertyDataset(
        config.dataset_path, config.target_name,
        edge_transform=edge_transform_func, heavy_only=config.heavy_only,
        split_file=config.split_file, split_type=split_type,
        model_pos_path=model_pos_path
    )
    return dset


def save_eval_scores(ckpt_path, file_name, results, tag, logger):
    # save scores
    meta_df = pd.DataFrame([results], index=[tag])
    df_path = os.path.join(ckpt_path, file_name)
    if os.path.exists(df_path):
        ori_df = pd.read_csv(df_path, index_col=0)
        meta_df = ori_df.append(meta_df)
    meta_df.to_csv(df_path)
    logger.info(f'Save eval results to {df_path} with tag {tag}')


class CheckpointManager(object):
    def __init__(self, save_dir, best_k=5, logger=BlackHole(), keep_n_ckpt=None):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.best_k = best_k
        self.ckpts = []
        self.logger = logger

        self.ckpt_path_queue = collections.deque()
        self.keep_n_ckpt = keep_n_ckpt

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

        for _ in range(max(self.best_k - len(self.ckpts), 0)):
            self.ckpts.append({
                'score': float('inf'),
                'file': None,
                'iteration': -1,
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None

    def _remove_extra_ckpts(self):
        if self.keep_n_ckpt is None:
            return
        while len(self.ckpt_path_queue) > self.keep_n_ckpt:
            path = self.ckpt_path_queue.popleft()
            if os.path.exists(path):
                os.remove(path)

    def _torch_save(self, data, path, logger=None):
        torch.save(data, path)
        self.ckpt_path_queue.append(path)
        if logger:
            logger.info(f'Model is saved to {path}')

    def save(self, model, opt, sche, config, score, step, logger=None):
        idx = self.get_worst_ckpt_idx()
        if idx is None:
            return False

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        self._torch_save({
            'config': config,
            'state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'sche_state_dict': sche.state_dict()
        }, path, logger)
        self._remove_extra_ckpts()

        self.ckpts[idx] = {
            'score': score,
            'file': fname,
            'iteration': step
        }

        return True

    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        self.logger.info(f'log dir: {self.save_dir}')
        self.logger.info(repr(self.ckpts[idx]))
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']), map_location=torch.device('cpu'))
        ckpt['iteration'] = self.ckpts[idx]['iteration']
        ckpt['score'] = self.ckpts[idx]['score']
        return ckpt

    def load_with_iteration(self, iteration):
        idx = None
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] == iteration:
                idx = i
        if idx is None:
            raise IOError('No checkpoints found.')
        self.logger.info(repr(self.ckpts[idx]))
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']), map_location=torch.device('cpu'))
        ckpt['iteration'] = self.ckpts[idx]['iteration']
        ckpt['score'] = self.ckpts[idx]['score']
        return ckpt

    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        self.logger.info(repr(self.ckpts[idx]))
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']), map_location=torch.device('cpu'))
        ckpt['iteration'] = self.ckpts[idx]['iteration']
        ckpt['score'] = self.ckpts[idx]['score']
        return ckpt
