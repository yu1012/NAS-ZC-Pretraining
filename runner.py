import os
import sys
import json
import time
import copy

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch_geometric.loader import DataLoader as GDataLoader

from utils import parse_args, seed_everything
from gnn import GIN_Predictor

import warnings
warnings.filterwarnings("ignore")

import wandb
torch.set_num_threads(1)

if __name__ == "__main__":
    seed_everything()
    args = parse_args()
    args.device = f'cuda:{args.gpu}'

    perms = np.load("./data/nats_permutation.npy")
    dataset = torch.load("./data/natsbench.pt")
    labels = torch.load("./data/nats_labels.pt")
    ranking = torch.load("./data/nats_ranking.pt")
    
    proxys = torch.stack([i.proxys[-1] for i in dataset])
    proxys -= proxys.min(0, keepdim=True)[0]
    proxys /= proxys.max(0, keepdim=True)[0]
    for i, data in enumerate(dataset):
        data.proxys = proxys[i]

    results = {}
    split = args.split
    perm = perms[split]
    wandb.init(
        project="NAS",
        name=f'{split}_{args.num_train}_{args.num_val}',
        config={
        "split": split,
        "pretrain": args.pretrain,
        "freeze": args.freeze,
        "pretrain-lr": args.pretraining_lr,
        "lr": args.lr,
        "train_data": args.num_train,
        "val_data": args.num_val
        }
    )

    predictor = GIN_Predictor(args, labels, device=args.device)

    if args.pretrain:
        if args.ranking_loss:
            masks, data = ranking_data(ranking, perms, dataset)
            predictor.ranking_fit(masks, data)
        else:
            masks, data = get_data(config, True, perm, dataset, config.dataset.batch_size, device)
            e_t_hist, e_v_hist = predictor.proxy_fit(masks, data)
        predictor.model.lin2 = nn.Linear(args.dim_hid, args.dim_out, True)
        
        # if args.freeze:
        #     for p in predictor.model.conv1.parameters():
        #         p.require_grads = False
        #     for p in predictor.model.convs.parameters():
        #         p.require_grads = False
        #     for p in predictor.model.lin1.parameters():
        #         p.require_grads = False

    predictor.model = predictor.model.to(device)
    masks, data = get_data(config, False, perm, dataset, config.dataset.batch_size, device)
    e_t_hist, e_v_hist = predictor.fit(masks, data)
    metrics = predictor.query(masks, data)

    wandb.log(metrics)
    wandb.finish()

    # for key in metrics:
    #     if key not in results:
    #         results[key] = [metrics[key]]
    #     else:
    #         results[key].append(metrics[key])

    # for key in results:
    #     print(key, ": {:.4f}({:.4f})".format(np.mean(results[key]), np.std(results[key])))

    # with open('./gen_results/{}.json'.format(method), 'w') as f:
    #     json.dump(results, f)