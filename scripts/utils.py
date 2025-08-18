import math
import torch
from random import Random
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
from sklearn import metrics
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from rdkit.Chem.Scaffolds import MurckoScaffold


def xavier_normal_small_init_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -a, a)




def evaluate(y_true, y_pred, y_smile, requirement, data_mean, data_std, data_task):
    collect_result = {}
    if 'sample' in requirement:
        collect_result['smile'] = y_smile.tolist()
        if data_task == 'classification':
            collect_result['prediction'] = y_pred.tolist()
            collect_result['label'] = y_true.tolist()
        else:
            collect_result['prediction'] = (y_pred * data_std + data_mean).tolist()
            collect_result['label'] = y_true.tolist()
    if 'rmse' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['rmse'] = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
    if 'mae' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'bce' in requirement:
        y_mask = np.where(y_true == -1, 0, 1)
        y_cal_true = np.where(y_true == -1, 0, y_true)
        loss = F.binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_cal_true), reduction='none') * y_mask
        collect_result['bce'] = loss.sum() / y_mask.sum()
    if 'auc' in requirement:
        auc_score_list = []
        if y_true.shape[1] > 1:
            for label in range(y_true.shape[1]):
                true, pred = y_true[:, label], y_pred[:, label]
                if len(set(true)) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    if len(set(true[np.where(true >= 0)])) !=1:
                        auc_score_list.append(metrics.roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
                    else:
                        auc_score_list.append(float('nan'))
            collect_result['auc'] = np.nanmean(auc_score_list)
        else:
            if len(set(y_true.reshape(-1))) == 1:
                collect_result['auc']=float('nan')
            else:
                collect_result['auc'] = metrics.roc_auc_score(y_true, y_pred)

    if 'spearman' in requirement:
        from scipy.stats import spearmanr
        y_t = y_true.flatten()
        y_p = (y_pred.flatten() * data_std + data_mean).tolist()
        rho, _ = spearmanr(y_t, y_p)
        collect_result['spearman'] = rho
    if 'smooth_l1' in requirement:
        y_true_t = torch.as_tensor(y_true).cpu()
        y_pred_t = torch.as_tensor(y_pred).cpu()
        collect_result['smooth_l1'] = float(
            F.smooth_l1_loss(y_pred_t, y_true_t, reduction='mean'))
    y_true_t = torch.as_tensor(y_true).cpu()
    y_pred_t = torch.as_tensor(y_pred).cpu()
    y_true_np = y_true_t.numpy()
    y_pred_np = y_pred_t.numpy()
    if 'auprc' in requirement:
        if y_true_np.ndim == 1:
            collect_result['auprc'] = (
                metrics.average_precision_score(y_true_np, y_pred_np)
                if len(np.unique(y_true_np)) > 1 else float('nan')
            )
        else:
            pr_list = []
            for t, p in zip(y_true_np.T, y_pred_np.T):
                valid = t != -1
                pr_list.append(
                    metrics.average_precision_score(t[valid], p[valid])
                    if len(np.unique(t[valid])) > 1 else float('nan')
                )
            collect_result['auprc'] = float(np.nanmean(pr_list))
    return collect_result


def scaffold_split(mol_list, frac=None, balanced=False, include_chirality=False, ramdom_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    random = Random(ramdom_state)

    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return train_index, valid_index, test_index


class ScheduledOptim:
    def __init__(self, optimizer, factor, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.factor = factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def view_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.factor * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def get_options(dataset_name):
    if dataset_name in ['esol', 'freesolv', 'caco2_wang', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb']:
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 3,
            'h': 4,
            'N_dense': 1,
            'n_generator_layers': 2,
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'aggregation_type': 'gru',
            'scale_norm': True,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'rmse' if dataset_name in ['esol', 'freesolv'] else 'mae',
            'metric': 'rmse' if dataset_name in ['esol', 'freesolv'] else 'mae',
            'task': 'regression',
            'num_workers': 4
        }

    elif dataset_name == 'lipophilicity':
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 3,
            'h': 8,
            'N_dense': 1,
            'n_generator_layers': 2,
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.15,
            'total_warmup_epochs': 30,
            'loss_function': 'rmse',
            'metric': 'rmse',
            'task': 'regression',
            'num_workers': 4
        }
    elif dataset_name in ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2047_EC50', 'CHEMBL2034_Ki', 'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 'CHEMBL4616_EC50', 'CHEMBL4792_Ki']:
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 3, 'h': 4, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1, 'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'aggregation_type': 'gru',
            'scale_norm': True, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150, 'batch_size': 32,
            'warmup_factor': 0.1, 'total_warmup_epochs': 30,
            'loss_function': 'rmse', 'metric': 'rmse',
            'task': 'regression', 'num_workers': 4
        }
    elif dataset_name in ['qm7', 'qm8', 'qm9', 'vdss_lombardo']:
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 6, 'h': 4, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1 if dataset_name == 'qm7' or dataset_name == 'vdss_lombardo' else 12 if dataset_name == 'qm8' else 3,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': True, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 10 if dataset_name in ['qm7', 'qm9'] else 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'mae' if dataset_name in ['qm7', 'qm8', 'qm9'] else 'smooth_l1',
            'metric': 'mae' if dataset_name in ['qm7', 'qm8', 'qm9'] else 'spearman',
            'task': 'regression',
            'num_workers': 4
        }
    elif dataset_name == 'bbbp':
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 4, 'h': 4, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.2,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }
    elif dataset_name == 'clintox':
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 4,
            'h': 4,
            'N_dense': 1,
            'n_generator_layers': 2,
            'n_output': 2,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.15,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }

    elif dataset_name in ['bace', 'bbb_martins', 'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'CHEMBL1613998', 'CHEMBL1614408', 'CHEMBL1614450', 'CHEMBL1738078', 'CHEMBL2219110', 'CHEMBL3215081', 'CHEMBL3888461', 'CHEMBL829401', 'CHEMBL900190']:
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 4, 'h': 8, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }
    elif dataset_name == 'clintox':
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 5,
            'h': 4,
            'N_dense': 1,
            'n_generator_layers': 2,
            'n_output': 2,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.15,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }

    elif dataset_name == 'sider':
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 4,
            'h': 4,
            'N_dense': 1,
            'n_generator_layers': 2,
            'n_output': 27,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'aggregation_type': 'gru',
            'scale_norm': True,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.2,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }
    elif dataset_name == 'zinc15_250K':
        model_params = {
            'd_atom': 115,
            'd_edge': 13,
            'd_model': 256,
            'N': 4,
            'h': 4,
            'N_dense': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'scale_norm': True,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32,
            'warmup_factor': 0.2,
            'total_warmup_epochs': 30,
            'num_workers': 4
        }
    elif dataset_name in ['tox21', 'toxcast']:
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 8, 'h': 4, 'N_dense': 1, 'n_generator_layers': 4,
            'n_output': 12 if dataset_name == 'tox21' else 617,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': True if dataset_name == 'toxcast' else False,
            'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150,
            'batch_size': 32 if dataset_name == 'tox21' else 16,
            'warmup_factor': 0.1,
            'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification',
            'num_workers': 4
        }
    elif dataset_name in ['cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_veith']:
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 8, 'h': 4, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1, 'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'softmax',
            'aggregation_type': 'gru',
            'scale_norm': False, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150, 'batch_size': 32,
            'warmup_factor': 0.1, 'total_warmup_epochs': 30,
            'loss_function': 'bce',
            'metric': 'auprc',
            'task': 'classification',
            'num_workers': 4,
        }
    elif dataset_name == 'hiv':
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 4, 'h': 8, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 1, 'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150, 'batch_size': 1,
            'warmup_factor': 0.1, 'total_warmup_epochs': 30,
            'loss_function': 'bce', 'metric': 'auc',
            'task': 'classification', 'num_workers': 0
        }
    elif dataset_name == 'muv':
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 4, 'h': 8, 'N_dense': 1, 'n_generator_layers': 2,
            'n_output': 17, 'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150, 'batch_size': 32,
            'warmup_factor': 0.1, 'total_warmup_epochs': 30,
            'loss_function': 'bce', 'metric': 'auc',
            'task': 'classification', 'num_workers': 4
        }
    elif dataset_name == 'nmrshiftdb':
        model_params = {
            'd_atom': 115, 'd_edge': 13, 'd_model': 256,
            'N': 8, 'h': 8, 'N_dense': 2, 'n_generator_layers': 4,
            'n_output': 1, 'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': None,
            'scale_norm': True, 'dropout': 0.0,
        }
        train_params = {
            'total_epochs': 150, 'batch_size': 32,
            'total_warmup_epochs': 30,
            'task': 'regression',
            'lr': 0.001, 'decay': 0, 'num_workers': 4
        }
    else:
        pass
    return model_params, train_params