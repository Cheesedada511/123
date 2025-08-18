import sys
sys.path.append('..')
import argparse
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_process.dataset_graph import pretrain_mol_collate_func,pretrain_construct_dataset
from models.prismnet import make_pretrain_model 
from scripts.utils import ScheduledOptim, get_options
from collections import defaultdict
import torch.nn as nn

def model_train(model, train_dataset, model_params, train_params, dataset_name,args):
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=pretrain_mol_collate_func,shuffle=True, drop_last=True, num_workers=0, pin_memory=False,prefetch_factor=2)
    node_criterion = nn.MSELoss()
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])

    if args.epochs !=None:
        train_params['total_epochs'] =args.epochs
    bestloss = float('inf') 

    for epoch in range(train_params['total_epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            num_nodes_list,smile_list, adjacency_matrices_list, node_features_list, edge_features_list, masked_labels_list = batch
            adjacency_matrices_list = [adj.to(train_params['device']) for adj in adjacency_matrices_list]
            node_features_list = [nf.to(train_params['device']) for nf in node_features_list] 
            edge_features_list = [ef.to(train_params['device']) for ef in edge_features_list]
            masked_labels_list = [ml.to(train_params['device']) for ml in masked_labels_list]
            batch_mask_list = [(torch.sum(torch.abs(nf), dim=-1) != 0) for nf in node_features_list]
            optimizer.zero_grad()
            reconstructed_node_features = model(node_features_list, batch_mask_list, adjacency_matrices_list, edge_features_list)
            mask_loss = torch.tensor(0.0, device=train_params['device'])
            global_loss = torch.tensor(0.0, device=train_params['device'])
            pred_features = reconstructed_node_features
            true_features = masked_labels_list[0]
            mask = (torch.sum(true_features, dim=-1) != 0)
            if mask.sum() > 0:
                pred_f = pred_features[mask]
                true_f = true_features[mask]
                mask_loss += node_criterion(pred_f, true_f).mean()
            for idx, num_nodes in enumerate(num_nodes_list):
                global_node_index = num_nodes-1
                pred_global_node_features = reconstructed_node_features[idx, global_node_index, :]
                true_global_node_features = node_features_list[0][idx, global_node_index, :]
                global_loss += node_criterion(pred_global_node_features, true_global_node_features)
            mask_loss += global_loss
            mask_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step_and_update_lr()
            epoch_loss += mask_loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        if(epoch_loss<bestloss):
            bestloss = epoch_loss
            torch.save(model.state_dict(), "../ckpts/pretrain.pth")
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=22)
    parser.add_argument("--gpu", type=str, help='gpu id', default=0)
    parser.add_argument("--dataset", type=str, help='pretraining dataset', default='zinc15_250K')
    parser.add_argument("--epochs", type=int, help="number of total epochs to run", default=3000)
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
        data_mol = pkl.load(f)

    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    dataset = pretrain_construct_dataset(data_mol, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])

    total_metrics = defaultdict(list)
    train_params['total_warmup_steps'] = \
        int(len(dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']

    model = make_pretrain_model(**model_params)
    model = model.to(train_params['device'])
    model_train(model, dataset, model_params, train_params, args.dataset, args)
