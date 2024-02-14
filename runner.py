import numpy as np
import time
import copy
import json
import os
import glob

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import StrainTensorNet
from utils.datamodule import get_loaders, get_eval_loader
from utils.etc import *

class Runner:
    def __init__(self, cfg):
        self.hparams = cfg.model.conv
        self.opt = cfg.otim.optimizer
        self.sch = cfg.otim.scheduler
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion_reg = self.loss_fn(cfg.model.task_loss)
        self.criterion_class = self.loss_fn('bce_loss')
        self.weight_class_loss = cfg.model.weight_class_loss

        self.num_teacher_forcing = cfg.model.num_teacher_forcing
        self.max_edge_size = cfg.dataset.max_edge_size
        self.bs = cfg.model.batch_size
        self.num_epochs = cfg.model.num_epochs
        self.graph_params = cfg.model.graph_params

    def _load_checkpoint(self, model, optimizer=None, scheduler=None, mode='train'):
        ckpts = list(glob.glob(f'{self.cfg.out_dir}/*={self.cfg.suffix}.ckpt'))  
        
        if len(ckpts) > 0:
            ckpt_epochs = [ckpt.split('/')[-1].split('.')[0].split('=')[1] for ckpt in ckpts]
            ckpt_epochs = np.array([int(epoch) if epoch != 'best' else -1 for epoch in ckpt_epochs])
            if mode == 'train':
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            elif mode == 'predict':
                ckpt = str(ckpts[ckpt_epochs.argsort()[0]])
            
            print(f">>>>> Load model from checkpoint:\n{ckpt}\n")
            if torch.cuda.is_available():
                ckpt = torch.load(ckpt)
            else:
                ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
            
            print(f">>>>> Update model from checkpoint:") 
            model.load_state_dict(ckpt['model_state_dict'])

            if mode == 'train':
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except:
                    print("ERROR: Loading optimizer")
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except:
                    print("ERROR: Loading scheduler")                
                   
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            best_val_epoch = ckpt['best_val_epoch']
            ckpt_path = f"{self.cfg.out_dir}/epoch={ckpt['epoch']}={self.cfg.suffix}.ckpt"
            best_ckpt_path = f"{self.cfg.out_dir}/epoch=best={self.cfg.suffix}.ckpt"
            print(f"revived epoch: {ckpt['epoch']}")
            print(f"best_val_reg_loss: {best_val_loss}")
            print(f"best_val_epoch: {best_val_epoch}\n")
            if mode == "train":
                print(">>>>> CONTINUE TRAINING")
            else:
                print(f">>>>> {mode.upper()}")

        else:
            print(f">>>>> NEW TRAINING")
            start_epoch = 0
            best_val_loss, best_val_epoch = 1e12, -1
            ckpt_path, best_ckpt_path = None, None
                
        return (model, optimizer, scheduler), start_epoch, (best_val_loss, best_val_epoch), (ckpt_path, best_ckpt_path)

    def _save_checkpoint(self, model_ckpt, ckpt_path, epoch):
        new_ckpt_path = f"{self.cfg.out_dir}/epoch={epoch}={self.cfg.suffix}.ckpt"
        if ckpt_path and os.path.exists(ckpt_path):
            os.remove(ckpt_path)        
        torch.save(model_ckpt, new_ckpt_path)
        #print(f"saved checkpoint at epoch [{epoch}]: ", new_ckpt_path)
        return new_ckpt_path

    def _get_checkpoint_dict(self, model, optimizer, scheduler, epoch, best_val_loss, best_val_epoch):
        model_ckpt = {
                'model_state_dict': copy.deepcopy(model.state_dict()), 
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()), 
                'scheduler_state_dict': copy.deepcopy(scheduler.state_dict()),
                'epoch': epoch, 
                'best_val_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
                #'train_reg_loss': log_dict['train_reg_loss'], 
                #'val_reg_loss': log_dict['val_reg_loss'] 
            }
        return model_ckpt

    def _logging(self, epoch, train_loss, train_acc, val_loss, val_acc):
        log_dict = {'epoch': epoch, 
                    'train_reg_loss': train_loss[0].item(), 
                    'train_class_loss': train_loss[1].item(), 
                    'train_tot_loss': train_loss[2].item(),
                    'train_acc0': train_acc[0].item(),
                    'train_acc1': train_acc[1].item(),
                    'train_acc': train_acc[2],
                    'val_reg_loss': val_loss[0].item(),
                    'val_class_loss': val_loss[1].item(),
                    'val_tot_loss': val_loss[2].item(),
                    'val_acc0': val_acc[0].item(),
                    'val_acc1': val_acc[1].item(),
                    'val_acc': val_acc[2],
                   }  
        
        with open(self.cfg.out_dir + "/training_metrics.json", "a") as f:
            results = {k: v for k, v in log_dict.items()}
            json.dump(results, f)
            f.write("\n")

    def _early_stopping(self, epoch, best_val_epoch):
        if epoch - best_val_epoch > self.cfg.early_stopping_patience:
            return True
        else:
            return False

    def loss_fn(self, task_loss):
        if task_loss == 'l1_loss':
            criterion = nn.L1Loss()
        elif task_loss == 'mse_loss':
            criterion = nn.MSELoss()
        elif task_loss == 'bce_loss':
            criterion = nn.BCELoss()
        return criterion

    def get_acc(self, label, gt_label, ng):
        acc = (label > 0.5).eq(gt_label).sum().item() / 21 / ng
        acc0 = ((label <= 0.5) & (gt_label == 0)).sum().item() / (gt_label == 0).sum().item()
        acc1 = ((label > 0.5) & (gt_label == 1)).sum().item() / (gt_label == 1).sum().item()
        return np.array([acc0, acc1, acc]) 

    def train_step(self, epoch, model, dataloader, optimizer, scheduler, train_size, df):
        model.train()
        avgloss, acc = 0., [0.,0.,0.]
        avgloss = np.zeros(3)
        num_iters = len(dataloader)
        ys, outs = [], []
        mpids, cijs, systems = [], [], []
        dataloader = iter(dataloader)
        teacher_forcing = True if epoch < self.num_teacher_forcing else False
        print(f"teacher: {teacher_forcing}")
        for i in range(num_iters):
            data = next(dataloader)
            g = data[0].to(self.device)
            y = data[1].to(self.device)
            mpid, ij, crystal_systems = data[2:] # the last item in the tuple is crystal systems.
            
            crystal_systems = df.loc[mpid]['spacegroup.crystal_system'].tolist()
            gt_label = ij_labels(ij, crystal_systems, 'torch').to(self.device)
            #ij_index = ij_labels(ij, ['triclinic']*bs, 'torch').to(self.device)
            ij_index = rand_ij_index(gt_label).to(self.device)
            assert gt_label.shape == ij_index.shape
    
            # run model forward and compute loss
            pred, label = model(g, ij_index, gt_label, teacher_forcing)
            loss_reg = self.criterion_reg(pred, y)
            loss_class = self.criterion_class(label, gt_label)
            loss = loss_reg + self.weight_class_loss * loss_class
            #rl = loss * std 
            #avgloss += loss * y.shape[0]
            avgloss += np.array([loss_reg.detach().item(), loss_class.detach().item(), loss.detach().item()]) * y.shape[0]
    
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ng = y.shape[0]
            correct = (label > 0.5).eq(gt_label).sum().item() / 21 / ng
            correct0 = ((label<=0.5) & (gt_label==0)).sum().item() / (gt_label==0).sum() 
            correct1 = ((label>0.5) & (gt_label==1)).sum().item() / (gt_label==1).sum() 
            acc[2] += correct * ng
            acc[0] += correct0 * ng
            acc[1] += correct1 * ng
            
            if i%100==0:
                pass #print(f"[{epoch}|{i}]: reg {loss_reg:.4f}, ml {loss_class:.4f}, tot {loss:.4f}, acc0 {correct0:.4f}, acc1 {correct1:.4f} acc {correct:.4f}")
            scheduler.step(epoch + i / num_iters)
            ys += y.cpu().tolist()
            outs += pred.cpu().tolist()
            mpids += mpid
            cijs += ij
            systems += crystal_systems
        
        avgloss /= train_size
        acc = [a/train_size for a in acc]
        #append_hist(ys, outs, avgloss, mpids, cijs, s_tracks, 'train')
        
        #print(f"[train|{epoch}]]: {avgloss.round(4)}, acc0 {acc[0]:.4f}, acc1 {acc[1]:.4f}, acc {acc[2]:.4f}")
        
        return avgloss, acc

    def eval_step(self, epoch, model, dataloader, eval_size, mode, df):
        model.eval()
        avgloss, acc = 0., [0.,0.,0.]
        avgloss = np.zeros(3)
        ys, outs = [], []
        mpids, cijs, systems = [], [], []
        num_iters = len(dataloader)
        dataloader = iter(dataloader)
        teacher_forcing = True if epoch < self.num_teacher_forcing else False
    
        with torch.no_grad(): 
            for i in range(num_iters):
                data = next(dataloader)
                g = data[0].to(self.device)
                y = data[1].to(self.device)
                mpid, ij, _ = data[2:] # the last item in the tuple is crystal systems.
                
                crystal_systems = df.loc[mpid]['spacegroup.crystal_system'].tolist()
                gt_label = ij_labels(ij, crystal_systems, 'torch').to(self.device)
                ij_index = ij_labels(ij, ['triclinic']*self.bs*2, 'torch').to(self.device)
                assert gt_label.shape == ij_index.shape
    
                # run model forward and compute loss
                pred, label = model(g, ij_index, gt_label, teacher_forcing)
                loss_reg = self.criterion_reg(pred, y)
                loss_class = self.criterion_class(label, gt_label)
                loss = loss_reg + self.weight_class_loss * loss_class
                #rl = loss * std
                #avgloss += loss * y.shape[0]
                avgloss += np.array([loss_reg.detach().item(), loss_class.detach().item(), loss.detach().item()]) * y.shape[0]
                ng = y.shape[0]
                correct = (label > 0.5).eq(gt_label).sum().item() / 21 / ng
                correct0 = ((label<=0.5) & (gt_label==0)).sum().item() / (gt_label==0).sum() 
                correct1 = ((label>0.5) & (gt_label==1)).sum().item() / (gt_label==1).sum() 
                acc[2] += correct * ng
                acc[0] += correct0 * ng
                acc[1] += correct1 * ng
   
                if i%100==0:
                    pass #print(f"...[{epoch}|{i}|{mode}]: reg {loss_reg:.4f}, ml {loss_class:.4f}, tot {loss:.4f}, acc0 {correct0:.4f}, acc1 {correct1:.4f}, acc {correct:.4f}")
                ys += y.cpu().tolist()
                outs += pred.cpu().tolist()
                mpids += mpid
                cijs += ij
                systems += crystal_systems 
    
            avgloss /= eval_size
            acc = [a/eval_size for a in acc]
            #append_hist(ys, outs, avgloss, mpids, cijs, s_tracks, mode)
           
            #print(f"...[{mode}|{epoch}]: {avgloss.round(4)}, acc0 {acc[0]:.4f}, acc1 {acc[1]:.4f}, acc {acc[2]:.4f}")
        return avgloss, acc

    def train(self, train_df, val_df):
        tic = time.perf_counter()

        print("LOAD DATASET")
        loaders, sizes, stats = get_loaders(train_df, val_df, self.max_edge_size, self.bs, self.graph_params)
        train_loader, val_loader = loaders
        train_size, val_size = sizes
        torch.save(stats, f'{self.cfg.out_dir}/stats_{self.cfg.suffix}.pt') # save mean and std
        print(f'saved directory: {self.cfg.out_dir}, saved name: {self.cfg.suffix}')
        
        model = StrainTensorNet(**self.hparams)
        optimizer = Adam(model.parameters(), **self.opt)
        scheduler = CosineAnnealingWarmRestarts(optimizer, **self.sch)
        
        models, start_epoch, bests, paths = self._load_checkpoint(model, 
                                                            optimizer, 
                                                            scheduler, 
                                                            'train')
        model, optimizer, scheduler = models
        best_val_reg_loss, best_val_epoch = bests
        ckpt_path, best_ckpt_path = paths
        print(model)
        model.to(self.device)

        for epoch in range(start_epoch, self.num_epochs):
            tic_epoch = time.perf_counter()
            
            train_loss, train_acc = self.train_step(epoch, model, train_loader, 
                                                    optimizer, scheduler, train_size, train_df)
            val_loss, val_acc = self.eval_step(epoch, model, val_loader, val_size, 'val', val_df)

            self._logging(epoch, train_loss, train_acc, val_loss, val_acc)  
            if val_loss[0] <= best_val_reg_loss:
                best_val_reg_loss = val_loss[0]
                best_val_epoch = epoch
                model_ckpt = self._get_checkpoint_dict(model, 
                                    optimizer, 
                                    scheduler, 
                                    epoch, 
                                    best_val_reg_loss, 
                                    best_val_epoch)
                best_ckpt_path = self._save_checkpoint(model_ckpt, best_ckpt_path, 'best')            
            model_ckpt = self._get_checkpoint_dict(model,
                                optimizer,
                                scheduler,
                                epoch, 
                                best_val_reg_loss, 
                                best_val_epoch)
            ckpt_path = self._save_checkpoint(model_ckpt, ckpt_path, epoch) 

            toc_epoch = time.perf_counter()
            self.epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc, 
                               best_val_reg_loss, best_val_epoch)
            print(f'{self.device} run time: {(toc_epoch - tic_epoch):.4f} s')
            print("----------\n")

            if self._early_stopping(epoch, best_val_epoch):
                print(f'Training has been early stopped at epoch {epoch}.')
                break

        toc = time.perf_counter()
        print(f'total {self.device} run time: {toc - tic} s')

    def to_6x6matrix(self, x):
        x = x.reshape(-1)
        assert x.shape[0] == 21
        matrix = np.zeros((6, 6))
        k = 0
        for i, j in zip(range(6), reversed(range(1,7))):
            matrix[i, i:] += x[k:k+j]
            k += j
        n, m = np.triu_indices(6, k=1)
        matrix[(m, n)] = matrix[(n, m)]
        return matrix

    def to_elastic_tensor(self, x, nsites, volume):
        unit_converter = 1000. / 160.2176487
        matrix = np.zeros((6, 6))
        const = nsites / ((self.graph_params.train.strain / 100) ** 2 * volume * unit_converter)
        for i in range(6):
            for j in range(i, 6):
                kron = 1. if i == j else 0.
                matrix[i, j] = const * (1 + kron) * (x[i, j] - (x[i, i] + x[j, j]) * (1 - kron))
                matrix[j, i] = matrix[i, j]
        return matrix

    @torch.no_grad()
    def predict(self, pred_df, limit_edge_size=False, use_gt_label=False):
        from tqdm import tqdm

        # Load model
        model = StrainTensorNet(**self.hparams)
        self._load_checkpoint(model, mode='predict')
        model.to(self.device)
        model.eval()

        # Dataloader
        if limit_edge_size:
            pred_df = pred_df[pred_df[f'{self.cfg.dataset.edge_style}_edge_size'] <= self.max_edge_size]
        dataloader = get_eval_loader(pred_df, self.cfg.batch_size, self.graph_params)

        # Etc
        mean, std = torch.load(f'{self.cfg.ckpt_dir}/stats_{self.cfg.suffix}.pt')
        uij, cij = {}, {}

        print("Start prediction")
        print(f"Number of crystal structures: {len(pred_df)}")
        for n, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Uij (meV/atom)'):
            g = data[0].to(self.device)
            mat_id, ij, _ = data[2:]
            ng = g.batch_size
            
            ij_index = ij_labels(ij, ['triclinic'] * ng, 'torch').to(self.device)
            forced_labels = None
            teacher_forcing = False
            
            if use_gt_label:
                if ("crystal_system" in pred_df.keys()) or ("spacegroup.crystal_system" in pred_df.keys()):
                    systems = pred_df.loc[mat_id]['spacegroup.crystal_system'].tolist()
                    gt_labels = ij_labels(ij, systems, 'torch').to(self.device)
                    forced_labels = gt_labels
                    teacher_forcing = True
                else:
                    print(f"use_gt_label is True, but crystal systems are not provided in the dataframe.")
                    break

            pred, label = model(g, ij_index, forced_labels, teacher_forcing)
            pred, label = pred.cpu().numpy(), label.cpu().numpy()
            if self.graph_params.train.normalize:
                pred = pred * std + mean

            pred = pred.reshape(-1)
            assert len(pred) == ng
            for k in range(ng):
                if mat_id[k] not in uij.keys():
                    uij[mat_id[k]] = {}
                uij[mat_id[k]][ij[k]] = pred[k]

        for n, key in enumerate(uij.keys()):
            u = np.array([uij[key][(i,j)] for i in range(6) for j in range(i,6)])
            u = self.to_6x6matrix(u)
            uij[key] = u
            mat_info = pred_df.iloc[n]
            cij[key] = self.to_elastic_tensor(u, mat_info['nsites'], mat_info['volume'])

        print("Finish prediction")
        return uij, cij

    def epoch_summary(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                      best_val_reg_loss, best_val_epoch):        
        print("----------")
        print(f">>>>> Summary epoch {epoch}:")
        print(f"Train loss [{epoch}]: reg {train_loss[0]:.4f}, class {train_loss[1]:.4f}, tot {train_loss[2]:.4f}")
        print(f"Val loss   [{epoch}]: reg {val_loss[0]:.4f}, class {val_loss[1]:.4f}, tot {val_loss[2]:.4f}") 
        print("----------")
        print(f"Train acc  [{epoch}]: <0> {train_acc[0]:.4f}, <1> {train_acc[1]:.4f}, all {train_acc[2]:.4f}")
        print(f"Val acc    [{epoch}]: <0> {val_acc[0]:.4f}, <1> {val_acc[1]:.4f}, all {val_acc[2]:.4f}")
        print("----------")
        print(f"Min val    [{best_val_epoch}]: reg {best_val_reg_loss:.4f}")
        
