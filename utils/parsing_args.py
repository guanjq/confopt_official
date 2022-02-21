import argparse
from collections import OrderedDict

from utils import misc as utils_misc


def add_conf_opt_args(parser):
    # data
    group_data = parser.add_argument_group('data')
    group_data.add_argument('--aux_edge_order', type=int, default=10)
    group_data.add_argument('--edge_transform_mode', type=str, default='full_edge',
                            choices=['aux_edge', 'full_edge', 'cutoff', 'none'])
    group_data.add_argument('--heavy_only', type=eval, default=False)
    group_data.add_argument('--cutoff', type=float, default=10.)
    group_data.add_argument('--cutoff_pos', type=str, default='rdkit_pos')
    group_data.add_argument('--dset_mode', type=str, default='relax_lowest',
                            choices=['lowest', 'relax_lowest', 'random_low', 'multi_low', 'multi_sample_low'])
    group_data.add_argument('--lowest_thres', type=float, default=1.0)
    # for sampling
    group_data.add_argument('--rdkit_pos_mode', type=str, default='online',
                            choices=['random', 'none', 'all', 'multi', 'online', 'online_ff'])

    # model
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--hidden_dim', type=int)
    group_model.add_argument('--num_blocks', type=int)
    group_model.add_argument('--num_layers', type=int)
    group_model.add_argument('--cutoff_mode', type=str, default='radius', choices=['radius', 'none'])
    group_model.add_argument('--ew_net_type', type=str, default='r', choices=['r', 'm', 'global', 'none'])
    group_model.add_argument('--r_feat_mode', type=str, default='sparse', choices=['basic', 'sparse'])
    group_model.add_argument('--energy_h_mode', type=str, default='basic',
                             choices=['basic', 'mix_distance', 'mix_cos_sim', 'share_distance', 'share_cos_sim'])
    group_model.add_argument('--x2h_out_fc', type=eval, default=False)
    group_model.add_argument('--sync_twoup', type=eval, default=False)
    group_model.add_argument('--norm', type=eval, default=True)

    # train
    group_train = parser.add_argument_group('train')
    group_train.add_argument('--propose_net_type', type=str, default='rdkit',
                             choices=['rdkit', 'random', 'gt', 'online_rdkit'])
    group_train.add_argument('--noise_type', type=str, default='const', choices=['const', 'expand'])
    group_train.add_argument('--noise_std', type=float, default=0.)
    group_train.add_argument('--batch_size', type=int, default=128)
    group_train.add_argument('--n_acc_batch', type=int, default=1)
    group_train.add_argument('--val_freq', type=int)
    group_train.add_argument('--train_report_iter', type=int)
    group_train.add_argument('--n_ref_samples', type=int, default=5)
    group_train.add_argument('--n_gen_samples', type=int, default=5)

    # for sampling
    group_train.add_argument('--loss_type', type=str, default='min', choices=['min', 'wasserstein'])
    parser.add_argument('--eval_propose_net_type', type=str, default='online_rdkit',
                        choices=['rdkit', 'random', 'online_rdkit'])
    parser.add_argument('--eval_noise', type=float, default=0.)

    # others (not saved to config)
    parser.add_argument('--logging', type=eval, default=True)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--keep_n_ckpt', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    return parser


def add_prop_pred_args(parser):
    # data
    group_data = parser.add_argument_group('data')
    group_data.add_argument('--target_name', type=str, default='homo', required=True)
    group_data.add_argument('--batch_size', type=int, default=128)
    group_data.add_argument('--n_acc_batch', type=int, default=1)
    group_data.add_argument('--edge_transform', type=str, default='none',
                            choices=['aux_edge', 'full_edge', 'cutoff', 'none'])
    group_data.add_argument('--cutoff_pos_key', type=str, default='pos')
    group_data.add_argument('--pre_pos_path', type=str, default=None)
    group_data.add_argument('--pre_pos_filename', type=str, default=None)

    # model
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--num_layers', type=int)
    group_model.add_argument('--hidden_dim', type=int)
    group_model.add_argument('--n_heads', type=int)
    group_model.add_argument('--num_r_gaussian', type=int)
    group_model.add_argument('--act_fn', type=str)
    group_model.add_argument('--pos_type', type=str, default=None, choices=['gt', 'random', 'rdkit', 'pre'])
    group_model.add_argument('--norm', type=eval, default=False)
    group_model.add_argument('--update_x', type=eval, default=False)
    group_model.add_argument('--cutoff_mode', type=str, default='none')
    group_model.add_argument('--ew_net_type', type=str, default='m')
    group_model.add_argument('--r_feat_mode', type=str, default='origin')
    group_model.add_argument('--energy_h_mode', type=str, default='basic')

    # Optimizer and scheduler
    group_train = parser.add_argument_group('train')
    group_train.add_argument('--lr', type=float, default=5e-4)
    group_train.add_argument('--sched_type', type=str, default='cos', choices=['plateau', 'cos'])
    group_train.add_argument('--lambda_pred_loss', type=float, default=10.)

    # Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_dir', type=str, default='./logs_prop_pred')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--train_report_iter', type=int, default=500)
    parser.add_argument('--keep_n_ckpt', type=int, default=5)
    return parser


def parse_config_args(config_args):
    config = utils_misc.load_config(config_args.config)
    defaults = {}
    defaults.update(config.data)
    defaults.update(config[config_args.model_type])
    defaults.update(config.train)
    if 'eval' in config:
        defaults.update(config.eval)
    return config, defaults


def parse_all_args(config, config_args, args):
    new_config = OrderedDict({'data': {}, 'model': {'model_type': config_args.model_type}, 'train': {}, 'eval': {}})
    new_config['model'].update(config[config_args.model_type])
    for arg in vars(args):
        if arg in config.data:
            new_config['data'][arg] = getattr(args, arg)
        elif arg in config[config_args.model_type]:
            new_config['model'][arg] = getattr(args, arg)
        elif arg in config.train:
            new_config['train'][arg] = getattr(args, arg)
        elif 'eval' in config and arg in config.eval:
            new_config['eval'][arg] = getattr(args, arg)

    for k, v in new_config.items():
        print(f'\n--[{k}]--')
        for argn, argv in v.items():
            print(f'[{argn}] {argv}')
    return new_config


def get_conf_opt_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, required=True)
    config_parser.add_argument('--model_type', type=str, default='ours_o2',
                               choices=['equi_se3trans', 'egnn', 'ours_o2', 'ours_o3'], required=True)
    config_args, remaining_argv = config_parser.parse_known_args()
    default_config, defaults = parse_config_args(config_args)

    parser = argparse.ArgumentParser()
    parser = add_conf_opt_args(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    new_config = parse_all_args(default_config, config_args, args)
    return args, new_config


def get_prop_pred_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, required=True)
    config_parser.add_argument('--model_type', type=str, default='ours_o2',
                               choices=['schnet', 'mpnn', 'egnn', 'ours_o2'], required=True)
    config_args, remaining_argv = config_parser.parse_known_args()
    default_config, defaults = parse_config_args(config_args)

    parser = argparse.ArgumentParser()
    parser = add_prop_pred_args(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    new_config = parse_all_args(default_config, config_args, args)
    return args, new_config
