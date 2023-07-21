import math
from tqdm import tqdm

import torch

from modules import GIN
from utils import evaluation
from predictor import Predictor
from model_utils import *

import wandb

class GIN_Predictor(Predictor):
    def __init__(self, params, labels=None, device=None, ss_type=None, encoding_type=None):
        super().__init__(labels, device, ss_type, encoding_type)
        self.params = params
        self.model = GIN(self.params.model.spec, task="graph").to(self.device)

    def proxy_fit(self, masks, loaders):
        train_loss_hist, val_loss_hist = self.gnn_proxy_fit(loaders["train"],
                                                          loaders["val"],loaders,masks,
                                                          self.params.model.pretraining)
        return train_loss_hist, val_loss_hist

    def pretrain_fit(self, masks, loaders):
        train_loss_hist, val_loss_hist = self.gnn_fit(loaders["train"],
                                                          loaders["val"],loaders,masks,
                                                          self.params.model.pretraining)
        return train_loss_hist, val_loss_hist

    def fit(self, masks, loaders):
        train_loss_hist, val_loss_hist = self.gnn_fit(loaders["train"],
                                                          loaders["val"],loaders,masks,
                                                          self.params.model.training)
        return train_loss_hist, val_loss_hist

    def ranking_fit(self, masks, loaders):
        train_loss_hist, val_loss_hist = self.gnn_ranking_fit(loaders["train"],
                                                          loaders["val"],loaders,masks,
                                                          self.params.model.pretraining)
        return train_loss_hist, val_loss_hist

    def query(self, masks, loaders):
        out = self.gnn_out(loaders["full"])
        metric = evaluation(out, self.labels, masks, rank_full=False)
        return metric

    def gnn_fit(self, train_loader, val_loader, loaders, masks, params):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        cnt = 0
        best_epoch = 0
        best_kt = 0 #math.inf
        best_val_model = self.model.state_dict()
        train_loss_hist = []
        val_loss_hist = []

        for epoch in tqdm(range(params.epoch)):
            train_loss = gnn_train(self.model, train_loader, optimizer, self.device,
                                   multi=False, loss_type=params.val_loss_type)
            train_loss_hist.append(train_loss)

            val_loss = gnn_eval(self.model, val_loader, self.device,
                                multi=False, loss_type=params.val_loss_type)
            val_loss_hist.append(val_loss)
            
            metric = self.query(masks, 1, loaders)
            wandb.log(metric)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch':epoch})
            
            if metric["kendall tau"] > best_kt:
                best_epoch = epoch
                best_kt = metric["kendall tau"]
                best_val_model = self.model.state_dict()
                cnt = 0
            cnt += 1
            if cnt > 75: break
        
        wandb.log({'best_kendall_tau': best_kt})
        self.model.load_state_dict(best_val_model)

        return train_loss_hist, val_loss_hist

    def gnn_out(self, loader):
        out = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data.x, data.edge_index, data.batch, embedding=False)
                out.append(output)
        out = torch.cat(out, dim=0)

        return out.to(self.device)

    def gnn_pretrain(self, train_loader, val_loader, loaders, masks, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)0

        cnt = 0
        best_epoch = 0
        best_kt = 0
        best_metric = 0
        best_val_model = self.model.state_dict()
        train_loss_hist = []
        val_loss_hist = []

        for epoch in tqdm(range(args.pretrain_epoch)):
            if epoch == 10:
                for p in self.model.conv1.parameters():
                    p.require_grads = True
                for p in self.model.convs.parameters():
                    p.require_grads = True
                for p in self.model.lin1.parameters():
                    p.require_grads = True

            train_loss = gnn_pre_train(self.model, train_loader,
                                       optimizer, self.device,
                                   multi=False, loss_type=params.val_loss_type)
            train_loss_hist.append(train_loss)

            val_loss = gnn_pre_eval(self.model, val_loader, self.device,
                                multi=False, loss_type=params.val_loss_type)
            val_loss_hist.append(val_loss)

            metric = self.query(masks, 1, loaders)
            wandb.log(metric)

            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch':epoch})

            if metric["kendall tau"] > best_kt:
                best_epoch = epoch
                best_kt = metric["kendall tau"]
                best_val_model = self.model.state_dict()
                best_metric = metric
                cnt = 0
            cnt += 1

            if cnt > 50:                    
                break

        for _ in range(epoch, params.epoch):
            wandb.log(best_metric)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch':epoch})

        # wandb.log({'best_kendall_tau': best_kt})
        # self.model.load_state_dict(best_val_model)

        return train_loss_hist, val_loss_hist

    
    def gnn_ranking_fit(self, train_loader, val_loader, loaders, masks, params):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=params.lr,
                                     weight_decay=params.weight_decay)

        cnt = 0
        best_epoch = 0
        best_val_loss = math.inf
        best_val_model = self.model.state_dict()
        train_loss_hist = []
        val_loss_hist = []
        for epoch in tqdm(range(params.epoch)):
            train_loss = gnn_ranking_train(self.model, train_loader,
                                       optimizer, self.device,
                                   multi=False, loss_type=params.val_loss_type)
            train_loss_hist.append(train_loss)

            val_loss = gnn_ranking_eval(self.model, val_loader, self.device,
                                multi=False, loss_type=params.val_loss_type)
            val_loss_hist.append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_model = self.model.state_dict()
                cnt = 0
            cnt += 1

            metric = self.query(masks, 1, loaders)
            wandb.log(metric)

            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch':epoch})
            
            if cnt > 20: break

        self.model.load_state_dict(best_val_model)

        return train_loss_hist, val_loss_hist

        