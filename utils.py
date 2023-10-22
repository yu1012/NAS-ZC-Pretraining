import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.stats import kendalltau, spearmanr, rankdata

import argparse

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--split', type=int, default=0, help='dataset split')
    parser.add_argument('--pretrain', action='store_true', help='whether pretrain or not')
    parser.add_argument('--freeze', action='store_true', help='whether freeze GIN or not on finetuning step')
    parser.add_argument('--ranking_loss', action='store_true', help='whether use ranking loss or not')
    parser.add_argument('--freeze_epoch', type=int, default=10, help='number of epochs to freeze GIN on finetuning step')
    parser.add_argument('--pretrain_epoch', type=int, default=50, help='training epoch of pretraining step')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch of finetuning step')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='learning rate of pretraining step')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dim_in', type=int, default=6, help='input dimension')
    parser.add_argument('--dim_hid', type=int, default=32, help='hidden dimension')
    parser.add_argument('--dim_out', type=int, default=1, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='number of GIN layers')
    parser.add_argument('--num_train', type=int, default=80, help='number of train samples')
    parser.add_argument('--num_val', type=int, default=20, help='number of validation samples')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    

    return parser.parse_args()

def seed_everything(seed: int = 29):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
        
def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def evaluation(out, labels, masks, rank_full=True):
    loss = F.mse_loss(out[masks["test"]],
                      labels[masks["test"]].reshape(-1, 1)).item()
    mae_loss = F.l1_loss(out[masks["test"]],
                         labels[masks["test"]].reshape(-1, 1)).item()

    r2 = r2_score(labels[masks["test"]].reshape(-1, 1).tolist(),
                  out[masks["test"]].tolist())
    if rank_full:
        pred_acc = out.reshape(-1).tolist()
        true_acc = labels.reshape(-1).tolist()
    else:
        pred_acc = np.array(out[masks["test"]].tolist()).reshape(-1)
        true_acc = np.array(labels[masks["test"]].reshape(-1).tolist())

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)

    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]

    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)

    # metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
    #           'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
    #           'p@10': precision_at_10, 'p@50': precision_at_50,
    #           'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}

    metric = {'knn test mse': loss, 'knn test mae': mae_loss,
              'kendall tau': tau, 'spearmanr coeff': coeff,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}

    return metric