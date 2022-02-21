import copy
from collections import defaultdict

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdchem import Mol

from models.conf_model import compute_min_loss, get_init_pos
from utils.conf import add_conformer, align_conformer, cal_rdkit_pos


def calculate_rmsd(mol, gt_pos, pos):
    # first decide to use the generated pos or the mirror pos
    # align conformer and calculate pos rms
    mol1 = copy.deepcopy(mol)
    mol1.RemoveAllConformers()
    mol2 = copy.deepcopy(mol1)
    if len(gt_pos.shape) == 3:
        add_conformer(mol1, gt_pos)
    else:
        add_conformer(mol1, np.expand_dims(gt_pos, 0))
    add_conformer(mol2, np.expand_dims(pos, 0))
    # rms1 = rdMolAlign.GetBestRMS(mol1, mol2)
    rms1 = min([rdMolAlign.GetBestRMS(mol2, mol1, refId=confID) for confID in range(mol1.GetNumConformers())])

    # consider the chiral case
    mirror_pos = copy.deepcopy(pos)
    mirror_pos[:, -1] *= -1
    mol2.RemoveAllConformers()
    add_conformer(mol2, np.expand_dims(mirror_pos, 0))
    # rms2 = rdMolAlign.GetBestRMS(mol1, mol2)
    rms2 = min([rdMolAlign.GetBestRMS(mol2, mol1, refId=confID) for confID in range(mol1.GetNumConformers())])
    if rms2 < rms1:
        pos = mirror_pos
        rms = rms2
    else:
        rms = rms1
    try:
        mol1 = Chem.RemoveHs(copy.deepcopy(mol1))
        mol2 = copy.deepcopy(mol)
        mol2.RemoveAllConformers()
        add_conformer(mol2, np.expand_dims(pos, 0))
        mol2 = Chem.RemoveHs(mol2)

        # all_heavy_rms = [rdMolAlign.GetBestRMS(mol1, mol2)]
        all_heavy_rms = [rdMolAlign.GetBestRMS(mol2, mol1, refId=confID) for confID in range(mol1.GetNumConformers())]
        heavy_rms = min(all_heavy_rms)
        match_gt_pos = gt_pos[np.argmin(all_heavy_rms)]
    except Chem.rdchem.AtomValenceException:
        heavy_rms = None
        match_gt_pos = gt_pos[0]
        # print('AtomValenceException remove Hs fail')
    return pos, match_gt_pos, rms, heavy_rms


def generate_mol_multi_confs(data, model, num_samples, rdkit_init_pos, val_batch_size, eval_noise, device):
    multi_gen_pos, multi_init_pos = [], []
    num_batch = int(np.ceil(num_samples / val_batch_size))
    for i in range(num_batch):
        batch_size = val_batch_size if i < num_batch - 1 else num_samples - i * val_batch_size
        batch = dgl.batch([data for _ in range(batch_size)])
        batch = batch.to(torch.device(device))
        fixed_init_pos = rdkit_init_pos[i * val_batch_size: (i * val_batch_size + batch_size)]
        fixed_init_pos = torch.from_numpy(fixed_init_pos).to(batch.ndata['node_feat']).view(-1, 3)
        fixed_init_pos += torch.randn_like(fixed_init_pos) * eval_noise

        with torch.no_grad():
            final_pos, _ = model(batch, fixed_init_pos)
        one_gen_pos = final_pos.view(batch_size, -1, 3)
        one_init_pos = fixed_init_pos.view(batch_size, -1, 3)

        one_gen_pos = one_gen_pos.cpu().numpy().astype(np.float64)  # (n_samples, sum_batch, 3)
        one_init_pos = one_init_pos.cpu().numpy().astype(np.float64)
        multi_gen_pos.append(one_gen_pos)
        multi_init_pos.append(one_init_pos)
    multi_gen_pos = np.concatenate(multi_gen_pos, axis=0)
    multi_init_pos = np.concatenate(multi_init_pos, axis=0)
    return multi_init_pos, multi_gen_pos


def generate_multi_confs(dset, model, eval_propose_net_type, val_batch_size, eval_noise, device,
                         heavy_only, ff_opt,
                         n_samples='auto', mode='model', return_gen_results=False, size_limit=None,
                         fix_init_pos=None):
    ref_mols, gen_mols = {}, {}
    all_gen_results = []
    assert dset.mode in ['relax_lowest', 'multi_low', 'multi_sample_low']
    if mode == 'model':
        model.eval()

    for data, labels, meta_info in tqdm.tqdm(dset, desc='Generate confs'):
        smiles, mol = meta_info['smiles'], meta_info['rdmol']

        ref_mol = copy.deepcopy(mol)
        gen_mol = copy.deepcopy(mol)
        ref_mol.RemoveAllConformers()
        gen_mol.RemoveAllConformers()
        ref_pos = labels.cpu().numpy().astype(np.float64)
        add_conformer(ref_mol, ref_pos)
        ref_mols[smiles] = ref_mol

        if n_samples == 'auto':
            num_samples = 2 * len(ref_mol.GetConformers())
        else:
            num_samples = n_samples

        if eval_propose_net_type == 'online_rdkit':
            init_pos, _ = cal_rdkit_pos(meta_info['ori_rdmol'], num_confs=num_samples,
                                        ff_opt=ff_opt, heavy_only=heavy_only, seed=-1)
        elif eval_propose_net_type == 'rdkit':
            init_pos = data.ndata['rdkit_pos'].permute(1, 0, 2).cpu().numpy().astype(np.float64)
            assert num_samples == init_pos.shape[0]
        elif eval_propose_net_type == 'random':
            init_pos = []
            n_nodes = data.ndata['node_feat'].shape[0]
            for _ in range(num_samples):
                init_pos.append(torch.randn(n_nodes, 3) * (1 + n_nodes * eval_noise))
            init_pos = torch.stack(init_pos, dim=0).numpy().astype(np.float64)
        elif eval_propose_net_type == 'fix':
            init_pos = fix_init_pos[smiles]
        else:
            raise ValueError(eval_propose_net_type)

        if mode == 'rdkit':
            add_conformer(gen_mol, init_pos)
            gen_mols[smiles] = gen_mol
            all_gen_results.append({'mol': meta_info['ori_rdmol'], 'gt_pos': ref_pos,  # [num_ref_confs, num_nodes, 3]
                                    'rdkit_pos': init_pos})
        elif mode == 'model':
            multi_init_pos, multi_gen_pos = generate_mol_multi_confs(
                data, model, num_samples, init_pos, val_batch_size, eval_noise, device)
            all_gen_results.append({'mol': meta_info['ori_rdmol'], 'gt_pos': ref_pos,  # [num_ref_confs, num_nodes, 3]
                                    'rdkit_pos': init_pos,
                                    'init_pos': multi_init_pos, 'gen_pos': multi_gen_pos})
            add_conformer(gen_mol, multi_gen_pos)
            gen_mols[smiles] = gen_mol
        if size_limit is not None and len(all_gen_results) >= size_limit:
            break
    if return_gen_results:
        return ref_mols, gen_mols, all_gen_results
    else:
        return ref_mols, gen_mols


def calculate_sampling_scores(ref_mols, gen_mols, delta, quick_mode):
    cov_list, mat_list, junk_list = [], [], []
    for smiles in tqdm.tqdm(ref_mols.keys(), desc='Align Confs'):
        ref_mol = ref_mols[smiles]
        gen_mol = gen_mols[smiles]
        if len(gen_mol.GetConformers()) == 0:
            continue
        # if ff_opt:
        #     AllChem.MMFFOptimizeMoleculeConfs(gen_mol, numThreads=0)

        cov, mat, junk, best_rms = align_conformer(
            ref_mol, gen_mol, heavy_only=True, delta=delta, quick_mode=quick_mode)
        cov_list.append(cov)
        mat_list.append(mat)
        junk_list.append(junk)

    results = {
        'cov_mean': np.mean(cov_list),
        'cov_median': np.median(cov_list),
        'mat_mean': np.mean(mat_list),
        'mat_median': np.median(mat_list),
        'junk_mean': np.mean(junk_list),
        'junk_median': np.median(junk_list),
    }
    return results


def validate_sampling_rdkit(dset, config, device, logger, quick_mode=True):
    ref_mols, gen_mols = generate_multi_confs(
        dset, None, config.eval.eval_propose_net_type, config.train.batch_size * 2, config.eval.eval_noise,
        device, config.data.heavy_only, config.eval.ff_opt,
        n_samples='auto', mode='rdkit', return_gen_results=False)
    r = calculate_sampling_scores(ref_mols, gen_mols, config.eval.delta, quick_mode)
    logger.info('[RDKit] Mean COV %.6f | Median COV %.6f | Mean MAT %.6f | Median MAT %.6f' % (
        r['cov_mean'], r['cov_median'], r['mat_mean'], r['mat_median']))

    return r['cov_mean'], r['cov_median'], r['mat_mean'], r['mat_median']


def validate_sampling_model(it, dset, val_loader, model, config, device, logger, prefix='Validate',
                            return_all_r=False, quick_mode=True):
    model.eval()
    r, _, _ = validate_scores(val_loader, model, device, config.data.cutoff, mode='model')
    loss = r['loss']
    ref_mols, gen_mols = generate_multi_confs(
        dset, model, config.eval.eval_propose_net_type, config.train.batch_size * 2, config.eval.eval_noise,
        device, config.data.heavy_only, config.eval.ff_opt,
        n_samples='auto', mode='model', return_gen_results=False)
    r = calculate_sampling_scores(ref_mols, gen_mols, config.eval.delta, quick_mode)
    logger.info('[%s] Iter %04d | Loss: %.6f | Mean COV %.6f | Median COV %.6f | Mean MAT %.6f | Median MAT %.6f' % (
        prefix, it, loss, r['cov_mean'], r['cov_median'], r['mat_mean'], r['mat_median']))

    if return_all_r:
        return r
    else:
        return loss


def validate_scores(val_loader, model, device, cutoff=10., mode='model'):
    all_gen_pos = []
    all_gen_results = []
    if mode == 'model':
        model.eval()

    sum_conf_loss, sum_n = 0, 0
    n_false_edges_pos, n_all_edges = 0, 0
    mse_by_order, n_by_order = defaultdict(list), defaultdict(list)
    mse_per_mol = []
    pos_rms_list, heavy_pos_rms_list = [], []
    success_list = []
    n_rmsd_fail = 0
    for batch, labels, batch_meta, labels_slices in tqdm.tqdm(
            val_loader, dynamic_ncols=True, desc='Validating', leave=None):
        batch = batch.to(torch.device(device))
        labels = labels.to(device)
        with torch.no_grad():
            if mode == 'model':
                init_pos = get_init_pos('rdkit', batch, labels, noise=0.,
                                        n_ref_samples=10, n_gen_samples=1, labels_slices=labels_slices, eval_mode=True)
                gen_pos, all_pos = model(batch, init_pos)

            elif mode == 'rdkit':
                gen_pos = batch.ndata['rdkit_pos']

        batch_conf_loss, batch_n, match_labels = compute_min_loss(
            batch, labels, gen_pos, labels_slices, n_gen_samples=1, return_match_labels=True)

        slices = np.cumsum([0] + batch.batch_num_nodes().tolist())
        l_slices = np.cumsum([0] + labels_slices)
        for idx, graph in enumerate(dgl.unbatch(batch)):
            pos = gen_pos[slices[idx]:slices[idx + 1]]
            # gt_pos = match_labels[slices[idx]:slices[idx + 1]]
            label = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, pos.shape[0], 3)
            label = label.cpu().numpy().astype(np.float64)
            meta_info = batch_meta[idx]
            smiles, mol, adj_order = meta_info['smiles'], meta_info['rdmol'], meta_info['adj_order']
            pos = pos.cpu().numpy().astype(np.float64)
            # gt_pos = gt_pos.cpu().numpy().astype(np.float64)
            pos, gt_pos, rms, heavy_rms = calculate_rmsd(mol, label, pos)
            pos_rms_list.append(rms)
            if heavy_rms is not None:
                heavy_pos_rms_list.append(heavy_rms)
            else:
                n_rmsd_fail += 1
            all_gen_pos.append(pos)

            # calculate distance scores
            pos = torch.from_numpy(pos).to(device)
            gt_pos = torch.from_numpy(gt_pos).to(device)
            gen_distmat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)
            gt_distmat = torch.norm(gt_pos.unsqueeze(0) - gt_pos.unsqueeze(1), p=2, dim=-1)
            # mse by order
            for order in range(1, int(adj_order.max())):
                index = (adj_order == order)
                mse = ((gen_distmat[index] - gt_distmat[index]) ** 2).sum()
                mse_by_order[order].append(mse)
                n_by_order[order].append(index.sum())
            mol_mse = F.mse_loss(gen_distmat, gt_distmat, reduction='none')
            mse_per_mol.append(mol_mse.mean())
            success_list.append(meta_info['rdkit_success'])

            full_adj_order = torch.where(adj_order == 0,
                                         (torch.ones_like(adj_order) - torch.eye(graph.num_nodes()).long()) * 10,
                                         adj_order)
            full_src, full_dst = torch.nonzero(full_adj_order > 0).t()
            full_r = torch.sqrt(torch.sum((gt_pos[full_dst] - gt_pos[full_src]) ** 2, -1, keepdim=True))
            edge_labels = (full_r < cutoff).float()
            # pred edges from pos
            pred_r = torch.sqrt(torch.sum((pos[full_dst] - pos[full_src]) ** 2, -1, keepdim=True))
            pred_edges_pos = (pred_r < cutoff).float()
            n_false_edges_pos += (pred_edges_pos - edge_labels).abs().sum()
            n_all_edges += len(edge_labels)

            if mode == 'model':
                all_b_pos = []
                for _, b_gen_pos in enumerate(all_pos):
                    all_b_pos.append(b_gen_pos[slices[idx]: slices[idx + 1]].cpu().numpy().astype(np.float64))

                all_gt_pos = labels[l_slices[idx]:l_slices[idx + 1]].view(-1, pos.shape[0], 3).cpu().numpy().astype(np.float64)
                all_gen_results.append({
                    'mol': meta_info['ori_rdmol'],
                    'gt_pos': all_gt_pos,  # [num_ref_confs, num_nodes, 3]
                    'conf_weights': meta_info['conf_weights'],
                    'init_pos': init_pos[slices[idx]:slices[idx + 1]].cpu().numpy().astype(np.float64),
                    'gen_pos': gen_pos[slices[idx]:slices[idx + 1]].cpu().numpy().astype(np.float64),
                    'all_pos': all_b_pos
                })

            if mode == 'rdkit':
                sum_conf_loss += mol_mse.mean()
                # sum_n += len(graph.nodes()) ** 2
                sum_n += 1

        if mode == 'model':
            sum_conf_loss += batch_conf_loss
            sum_n += batch_n

    pos_rms_array, heavy_pos_rms_array, success_array = np.asarray(pos_rms_list), np.asarray(
        heavy_pos_rms_list), np.asarray(success_list)
    avg_conf_loss = sum_conf_loss / sum_n
    avg_mse_by_order = {}
    for order, v in mse_by_order.items():
        avg_mse_by_order[order] = sum(v) / sum(n_by_order[order])
    avg_mse_per_mol = sum(mse_per_mol) / len(mse_per_mol)
    avg_pos_rms = pos_rms_array.mean()
    avg_heavy_pos_rms = heavy_pos_rms_array.mean()
    median_pos_rms = np.median(pos_rms_array)
    median_heavy_pos_rms = np.median(heavy_pos_rms_array)
    # separately compute rdkit succeed and fail case
    avg_success_pos_rms = pos_rms_array[success_array == 1].mean()
    avg_fail_pos_rms = pos_rms_array[success_array == 0].mean() if sum(success_array) < len(success_array) else 0.
    # print('num heavy rmsd calculation fail: ', n_rmsd_fail)
    results = {
        'loss': avg_conf_loss,
        'mse_by_order': avg_mse_by_order,
        'mse_per_mol': avg_mse_per_mol,
        'mean_rmsd': avg_pos_rms,
        'heavy_mean_rmsd': avg_heavy_pos_rms,
        'median_rmsd': median_pos_rms,
        'heavy_median_rmsd': median_heavy_pos_rms,
        'success_mean_rmsd': avg_success_pos_rms,
        'fail_mean_rmsd': avg_fail_pos_rms,
        'success_std_rmsd': pos_rms_array[success_array == 1].std(),
        'fail_std_rmsd': pos_rms_array[success_array == 0].std(),
        'n_success': sum(success_array == 1),
        'n_fail': sum(success_array == 0),
        'edge_false_rate_pos': n_false_edges_pos / n_all_edges
    }
    return results, all_gen_pos, all_gen_results


def validate_rdkit(val_loader, logger, device, cutoff=10., writer=None):
    r, _, _ = validate_scores(val_loader, None, device, cutoff, mode='rdkit')
    logger.info('[RDKit] Loss %.6f | MSE per mol %.6f' % (r['loss'], r['mse_per_mol']))
    for k, v in r['mse_by_order'].items():
        logger.info(f'[RDKit] {k}-order mse: {v:.6f}')
        if writer:
            writer.add_scalar(f'val/dist_{k}_order', v, 0)

    logger.info('[RDKit] Mean RMSD %.6f | Heavy Mean RMSD %.6f | Median RMSD %.6f | Median Heavy RMSD %.6f' % (
        r['mean_rmsd'], r['heavy_mean_rmsd'], r['median_rmsd'], r['heavy_median_rmsd']))
    logger.info('[RDKit] Edge False Rate Pos %.6f' % r['edge_false_rate_pos'])

    if writer:
        writer.add_scalar('val/loss', r['loss'], 0)
        writer.add_scalar('val/edge_false_rate_pos', r['edge_false_rate_pos'], 0)
        writer.add_scalar('val/heavy_mean_rmsd', r['heavy_mean_rmsd'], 0)
        writer.add_scalar('val/heavy_median_rmsd', r['heavy_median_rmsd'], 0)
        writer.flush()
    return r['loss'], r['mse_per_mol'], r['mean_rmsd'], r['heavy_mean_rmsd']


def validate_model(it, val_loader, model, logger, device, cutoff=10., writer=None, prefix='Validate',
                   return_all_r=False, return_all_gen_results=False):
    model.eval()
    r, _, all_gen_results = validate_scores(val_loader, model, device, cutoff, mode='model')
    logger.info('[%s] Iter %04d | Loss: %.6f | Edge false rate: Pos %.6f' % (
        prefix, it, r['loss'], r['edge_false_rate_pos']))
    logger.info('[%s] Mean RMSD %.6f | Heavy Mean RMSD %.6f | Median RMSD %.6f | Median Heavy RMSD %.6f' % (
        prefix, r['mean_rmsd'], r['heavy_mean_rmsd'], r['median_rmsd'], r['heavy_median_rmsd']))

    if writer:
        writer.add_scalar('val/loss', r['loss'], it)
        writer.add_scalar('val/edge_false_rate_pos', r['edge_false_rate_pos'], it)
        writer.add_scalar('val/heavy_mean_rmsd', r['heavy_mean_rmsd'], it)
        writer.add_scalar('val/heavy_median_rmsd', r['heavy_median_rmsd'], it)
        writer.flush()
    if return_all_gen_results:
        return all_gen_results
    elif return_all_r:
        return r
    else:
        return r['loss']
