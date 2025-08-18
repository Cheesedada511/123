import sys
sys.path.append('..')
import argparse
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_process.dataset_graph import construct_dataset, mol_collate_func
from models.prismnet import make_model 
from scripts.utils import ScheduledOptim, get_options, evaluate, scaffold_split
from collections import defaultdict

try:
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r 

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
 
    V = 2 * V.view(*x_shape)
    return V



def cal_loss(y_true, y_pred, loss_name, criterion, device,graph_features=None):
    if loss_name == 'smooth_l1':
        if isinstance(criterion, RegressionLoss):
            return criterion.compute(y_pred, y_true, graph_features=graph_features)
        else:
            raise ValueError("error")
    if loss_name == 'bce':
        y_true = y_true.long()
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true).float()
        if isinstance(criterion, ClassificationLoss):
            return criterion.compute(y_pred, y_cal_true, graph_features=graph_features)
        else:
            loss = criterion(y_pred, y_cal_true) * y_mask
            return loss.sum() / y_mask.sum()
    elif loss_name == 'mse':
        if isinstance(criterion, RegressionLoss):
            return criterion.compute(y_pred, y_true, graph_features=graph_features)
        else:
            return F.mse_loss(y_pred, y_true)
    elif loss_name == 'rmse':
        if isinstance(criterion, RegressionLoss):
            mse_loss = criterion.compute(y_pred, y_true, graph_features=graph_features)
        else:
            mse_loss = F.mse_loss(y_pred, y_true)
        return torch.sqrt(mse_loss)
    elif loss_name == 'mae':
        if isinstance(criterion, RegressionLoss):
            return criterion.compute(y_pred, y_true, graph_features=graph_features)
        else:
            return F.l1_loss(y_pred, y_true)
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}")

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss, dim=1)
    return focal_loss

class ClassificationLoss:
    def __init__(self, device, num_tasks, is_multitask=True, beta=0.9999, contrastive_weight=0.1, gamma=0.5, epsilon=1e-8):
        self.device = device
        self.num_tasks = num_tasks
        self.is_multitask = is_multitask
        self.task_weights = {}
        self.task_dynamic_weights = torch.ones(num_tasks, device=self.device, requires_grad=False)
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.contrastive_weight = contrastive_weight
        self.history_loss = torch.zeros(num_tasks).to(device)
        self.loss_accumulator = torch.zeros(num_tasks).to(device)
        self.loss_count = torch.zeros(num_tasks).to(device)

    def update_task_weights(self, train_labels):
        for task in range(self.num_tasks):
            task_labels = train_labels[:, task]
            valid_mask = task_labels != -1
            task_labels = task_labels[valid_mask]

            train_size = torch.bincount(task_labels.long())
            train_size_arr = train_size.cpu().numpy()
            train_size_mean = np.mean(train_size_arr)
            train_size_factor = train_size_mean / train_size_arr

            self.task_weights[task] = {
                'factor_train': torch.from_numpy(train_size_factor).float().to(self.device),
                'cls_num': len(train_size)
            }

            beta = self.beta
            effective_num = 1.0 - np.power(beta, train_size_arr)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(train_size)

            self.task_weights[task]['weights'] = torch.tensor(weights).float().to(self.device)
    def compute(self, pred, target, graph_features=None):
        total_loss_list = []
        contrastive_loss_list = []
        valid_tasks = 0
        task_losses = torch.zeros(self.num_tasks).to(self.device)

        for task in range(self.num_tasks):
            task_pred = pred[:, task] if pred.shape[1] > 1 else pred.squeeze(-1)
            task_target = target[:, task] if target.shape[1] > 1 else target.squeeze(-1)
            valid_mask = task_target != -1
            if valid_mask.sum() == 0:
                continue

            task_pred = task_pred[valid_mask]
            task_target = task_target[valid_mask]
            labels_one_hot = F.one_hot(task_target.long(), self.task_weights[task]['cls_num']).float()
            batch_size = task_pred.shape[0]

            if len(task_pred.shape) == 1:
                task_pred = task_pred.view(batch_size, 1)
            task_pred = torch.cat([1 - task_pred, task_pred], dim=1)

            weights = self.task_weights.get(task, {}).get('weights', torch.ones(2).to(self.device))
            weights = weights.unsqueeze(0).repeat(batch_size, 1)
            pred_softmax = torch.sigmoid(task_pred)

            loss = F.binary_cross_entropy(pred_softmax, labels_one_hot, weight=weights, reduction='mean')
            total_loss_list.append(loss * self.task_dynamic_weights[task].clone().detach())
            task_losses[task] = loss
            valid_tasks = valid_tasks + 1

            if graph_features is not None:
                task_labels = target[:, task] if target.shape[1] > 1 else target.squeeze(-1)
                task_contrastive_loss = self._compute_contrastive_loss(graph_features, task_labels) * self.task_dynamic_weights[task].clone().detach()
                contrastive_loss_list.append(task_contrastive_loss)

        self._update_dynamic_weights(task_losses)
        if len(total_loss_list) > 0:
            total_loss = torch.stack(total_loss_list).sum() / valid_tasks
        else:
            total_loss = torch.tensor(0.0, device=self.device)

        if len(contrastive_loss_list) > 0:
            contrastive_loss = torch.stack(contrastive_loss_list).sum()
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
        return total_loss + self.contrastive_weight * contrastive_loss

    def _compute_contrastive_loss(self, features, labels):
        valid_mask = labels != -1
        features = features[valid_mask]
        labels = labels[valid_mask]

        if labels.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrastive_features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(contrastive_features, contrastive_features.T)
        assert torch.isfinite(similarity_matrix).all(), "Similarity matrix contains NaN or Inf"
        logits = similarity_matrix / 0.5
        logits_max = logits.max(dim=1, keepdim=True).values
        logits = logits.clone() - logits_max.detach()
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        assert torch.isfinite(log_prob).all(), "Log-probabilities contain NaN or Inf"
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

    def _update_dynamic_weights(self, current_loss):
        with torch.no_grad():
            for task in range(self.num_tasks):
                if current_loss[task] == 0:
                    continue

                L_i = current_loss[task]
                self.loss_accumulator[task] += L_i
                self.loss_count[task] += 1
                L_bar = self.loss_accumulator[task] / self.loss_count[task]

                ratio = (L_i - L_bar) / (L_bar + self.epsilon)
                update = 1 + self.gamma * ratio
                self.task_dynamic_weights[task] *= update
                self.task_dynamic_weights[task] = self.task_dynamic_weights[task].clamp(min=0.1, max=10)
                self.history_loss[task] = L_i
        total_weight = self.task_dynamic_weights.sum()
        self.task_dynamic_weights /= total_weight

class RegressionLoss:
    def __init__(self, device, num_tasks, contrastive_weight=0.1, beta=0.9999, gamma=0.5, epsilon=1e-8):
        self.device = device
        self.num_tasks = num_tasks
        self.contrastive_weight = contrastive_weight
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.task_dynamic_weights = torch.ones(num_tasks, device=self.device, requires_grad=False)
        self.history_loss = torch.zeros(num_tasks, device=self.device, requires_grad=False)
        self.loss_accumulator = torch.zeros(num_tasks).to(device)
        self.loss_count = torch.zeros(num_tasks).to(device)

    def compute(self, pred, target, graph_features=None):
        total_loss_list = []
        contrastive_loss_list = []
        valid_tasks = 0
        task_losses = torch.zeros(self.num_tasks).to(self.device)

        for task in range(self.num_tasks):
            task_pred = pred[:, task]
            task_target = target[:, task]
            valid_mask = ~torch.isnan(task_target)

            if valid_mask.sum() == 0:
                continue

            task_pred = task_pred[valid_mask]
            task_target = task_target[valid_mask]

            loss = F.mse_loss(task_pred, task_target)
            total_loss_list.append(loss * self.task_dynamic_weights[task].clone().detach())
            task_losses[task] = loss
            valid_tasks += 1

            if graph_features is not None:
                task_contrastive_loss = self._compute_contrastive_loss(graph_features, task_target) * \
                                        self.task_dynamic_weights[task].clone().detach()
                contrastive_loss_list.append(task_contrastive_loss)

        self._update_dynamic_weights(task_losses)

        total_loss = torch.stack(total_loss_list).sum() / valid_tasks if total_loss_list else torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.stack(contrastive_loss_list).sum() if contrastive_loss_list else torch.tensor(0.0, device=self.device)

        return total_loss + self.contrastive_weight * contrastive_loss

    def _compute_contrastive_loss(self, features, labels):
        valid_mask = ~torch.isnan(labels)
        features = features[valid_mask]
        labels = labels[valid_mask]

        if labels.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        labels = labels.view(-1, 1)
        pairwise_diff = torch.abs(labels - labels.T)
        similarity_matrix = torch.exp(-pairwise_diff / 0.1)

        contrastive_features = F.normalize(features, dim=1)
        logits = torch.matmul(contrastive_features, contrastive_features.T)
        logits_max = logits.max(dim=1, keepdim=True).values
        logits = logits.clone() - logits_max.detach()

        exp_logits = torch.exp(logits) * similarity_matrix
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (similarity_matrix * log_prob).sum(1) / (similarity_matrix.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss

    def _update_dynamic_weights(self, current_loss):
        with torch.no_grad():
            for task in range(self.num_tasks):
                if current_loss[task] == 0:
                    continue
                L_i = current_loss[task]
                self.loss_accumulator[task] += L_i
                self.loss_count[task] += 1
                L_bar = self.loss_accumulator[task] / self.loss_count[task]
                ratio = (L_i - L_bar) / (L_bar + self.epsilon)
                update = 1 + self.gamma * ratio
                self.task_dynamic_weights[task] *= update
                self.task_dynamic_weights[task] = self.task_dynamic_weights[task].clamp(min=0.1, max=10)
                self.history_loss[task] = L_i
        total_weight = self.task_dynamic_weights.sum()
        self.task_dynamic_weights /= total_weight



def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, args):
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,shuffle=True, drop_last=True, num_workers=train_params['num_workers'], pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,shuffle=True, drop_last=True, num_workers=train_params['num_workers'], pin_memory=True)

    if train_params['task'] == 'classification':
        criterion = ClassificationLoss(
            device=train_params['device'],
            num_tasks=model_params['n_output'],
            is_multitask=model_params['n_output'] > 1
        )
        
        train_labels = []
        for batch in train_loader:
            _, _, _, _, y_true = batch
            if len(y_true.shape) == 1:
                y_true = y_true.unsqueeze(-1)
            train_labels.append(y_true)
        train_labels = torch.cat(train_labels, dim=0)
        criterion.update_task_weights(train_labels)
    else:
        criterion = RegressionLoss(
            device=train_params['device'],
            num_tasks=model_params['n_output'],
            contrastive_weight=0.1
        )
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])

    best_valid_metric = float('inf') if train_params['task'] == 'regression' else float('-inf')
    if args.epochs !=None:
        train_params['total_epochs'] =args.epochs
    for epoch in range(train_params['total_epochs']):
        train_loss = list()
        model.train()

        for batch in tqdm(train_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])
            node_features = node_features.to(train_params['device'])
            edge_features = edge_features.to(train_params['device'])
            y_true = y_true.to(train_params['device'])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            y_pred, out_pooling = model(node_features, batch_mask, adjacency_matrix, edge_features)

            if len(y_pred.shape) == 1:
                y_pred = y_pred.view(-1, 1)
            assert y_pred.shape[0] == y_true.shape[0], \
                f"Batch size mismatch: pred {y_pred.shape}, target {y_true.shape}"
            
            loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,train_params['device'],graph_features=out_pooling)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            train_loss.append(loss.detach().item())
        model.eval()
        with torch.no_grad():
            valid_true, valid_pred, valid_smile= list(), list(), list()
            for batch in tqdm(valid_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])
                node_features = node_features.to(train_params['device'])
                edge_features = edge_features.to(train_params['device'])
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

                y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)

                y_true = y_true.numpy()
                y_pred = y_pred.detach().cpu().numpy()

                valid_true.append(y_true)
                valid_pred.append(y_pred)
                valid_smile.append(smile_list)

            valid_true, valid_pred = np.concatenate(valid_true, axis=0), np.concatenate(valid_pred, axis=0)
            valid_smile= np.concatenate(valid_smile, axis=0)

        valid_result = evaluate(valid_true, valid_pred, valid_smile,
                                requirement=['sample', train_params['loss_function'], train_params['metric']],
                                data_mean=0, data_std=1, data_task=train_params['task'])

        if train_params['task'] == 'regression':
            if valid_result[train_params['metric']] < best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                torch.save({'state_dict': model.state_dict()},f'../ckpts/{dataset_name}.pt')
            print("Epoch {},  learning rate {}, "
                  "train loss: {}, "
                  "valid loss: {}, "
                  "best valid {}: {}"
                  .format(epoch + 1, optimizer.view_lr(),
                          np.mean(train_loss),
                          valid_result[train_params['loss_function']],
                          train_params['metric'], best_valid_metric
                          ))
        else:
            if valid_result[train_params['metric']] > best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                torch.save({'state_dict': model.state_dict()},f'../ckpts/{dataset_name}.pt')
            print("Epoch {},  learning rate {}, "
                  "train loss: {}, "
                  "valid loss: {}, "
                  "best valid {}: {}"
                  .format(epoch + 1, optimizer.view_lr(),
                          np.mean(train_loss),
                          valid_result[train_params['loss_function']],
                          train_params['metric'], best_valid_metric
                          ))

def model_test(checkpoint, test_dataset, model_params, train_params):
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,shuffle=False, drop_last=True, num_workers=train_params['num_workers'], pin_memory=True)
    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_smile = list(), list(), list()
        for batch in tqdm(test_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])
            node_features = node_features.to(train_params['device'])
            edge_features = edge_features.to(train_params['device'])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
            y_true = y_true.numpy()
            y_pred = y_pred.detach().cpu().numpy()

            test_true.append(y_true)
            test_pred.append(y_pred)
            test_smile.append(smile_list)
        test_true, test_pred = np.concatenate(test_true, axis=0), np.concatenate(test_pred, axis=0)
        test_smile= np.concatenate(test_smile, axis=0)
    test_result = evaluate(test_true, test_pred, test_smile,
                           requirement=['sample', train_params['loss_function'], train_params['metric']],
                           data_mean=0, data_std=1, data_task=train_params['task'])

    print("test {}: {}".format(train_params['metric'], test_result[train_params['metric']]))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu id', default="0")
    parser.add_argument("--dataset", type=str, help='fine-tuning dataset', default='toxcast')
    parser.add_argument("--type", type=str, help='choose a type', default='train',choices=['train', 'evaluate'])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, help="number of total epochs to run", default=None)
    args = parser.parse_args()

    model_params, train_params = get_options(args.dataset)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    with open(f'../data/{args.dataset}/{args.dataset}.pickle', 'rb') as f:
        [data_mol, data_label] = pkl.load(f)
    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])+1
    dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    total_metrics = defaultdict(list)

    train_index, valid_index, test_index = scaffold_split(data_mol, frac=[0.8, 0.1, 0.1], balanced=True,
                                                            include_chirality=False, ramdom_state=args.seed)
    train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

    train_params['total_warmup_steps'] = \
        int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']

    if train_params['task'] == 'regression':
        train_params['mean'] = np.mean(np.array(data_label)[train_index])
        train_params['std'] = np.std(np.array(data_label)[train_index])
    else:
        train_params['mean'], train_params['std'] = 0, 1

    model = make_model(**model_params)
    model = model.to(train_params['device'])
    if args.type =='train':
        ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])
        if 'pos_embed.pe.weight' in ckpt:
            del ckpt['pos_embed.pe.weight']
        model.load_state_dict(ckpt, strict=False)
        print(f'Successful Loading the ckpt ')
        print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
        model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args)
    else:
        ckpt = torch.load(args.ckpt_path, map_location=train_params['device'])
        model_test(ckpt, test_dataset, model_params, train_params)