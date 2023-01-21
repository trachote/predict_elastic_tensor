import numpy as np
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ssl import SE3Net
from utils.dataloader import get_loaders
from utils.etc import *

class Runner:
    def __init__(self, cfg, verbose=True):
        hparams = cfg.model.conv
        opt = cfg.otim.optimizer
        sch = cfg.otim.scheduler

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SE3Net(hparams.num_layers,
                            hparams.atom_feat_size,
                            hparams.num_channels,
                            num_nlayers=hparams.num_nlayers,
                            num_degrees=hparams.num_degrees,
                            edge_dim=hparams.num_bonds,
                            div=hparams.div,
                            n_heads=hparams.num_heads,
                            pooling=hparams.pooling,
                            embed_dim=hparams.embed_dim,
                            mid_dim=hparams.radial_dim)
        self.model = self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                     sch.T_0,
                                                     T_mult=sch.T_mult,
                                                     eta_min=sch.eta_min)

        self.criterion_reg = self.loss_fn(cfg.model.task_loss)
        self.criterion_class = self.loss_fn('bce_loss')
        self.weigth_class_loss = cfg.model.weigth_class_loss

        self.num_teacher_forcing = cfg.model.num_teacher_forcing
        self.max_edge = cfg.model.max_edge
        self.bs = cfg.model.batch_size
        self.num_epochs = cfg.model.num_epochs
        self.graph_params = cfg.model.graph_params

        if verbose:
            print(self.model)

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

    def train_step(self, epoch, dataloader, train_size):
        avgloss, acc_tot = np.zeros(3), np.zeros(3)
        num_iters = len(dataloader)
        dataloader = iter(dataloader)
        teacher_forcing = True if epoch < self.num_teacher_forcing else False

        self.model.train()
        for i in range(num_iters):
            data = next(dataloader)
            g = data[0].to(self.device)
            y = data[1].to(self.device)
            mpid, ij, systems = data[2:]
            ng = g.batch_size 

            gt_label = ij_labels(ij, systems, 'torch').to(self.device)
            ij_index = rand_ij_index(gt_label).to(self.device)
            assert gt_label.shape == ij_index.shape

            pred, label = self.model(g, ij_index, gt_label, teacher_forcing)
            loss_reg = self.criterion_reg(pred, y)
            loss_class = self.criterion_class(label, gt_label)
            loss = loss_reg + self.weigth_class_loss * loss_class
            avgloss += np.array([loss_reg.detach().item(), loss_class.detach().item(), loss.detach().item()]) * ng
            acc = self.get_acc(label, gt_label, ng)
            acc_tot += acc * ng

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + i / num_iters)

            if i%10 == 0:
                print(f"[{epoch}|{i}]: reg {loss_reg:.4f}, ml {loss_class:.4f}, tot {loss:.4f}, acc0 {acc[0]:.4f}, acc1 {acc[1]:.4f} acc {acc[2]:.4f}")

        avgloss /= train_size
        acc /= train_size
        return avgloss, acc_tot

    def eval_step(self, epoch, dataloader, eval_size, mode):
        avgloss, acc_tot = np.zeros(3), np.zeros(3)
        num_iters = len(dataloader)
        dataloader = iter(dataloader)
        teacher_forcing = True if epoch < self.num_teacher_forcing else False

        self.model.eval()
        with torch.no_grad():
            for i in range(num_iters):
                data = next(dataloader)
                g = data[0].to(self.device)
                y = data[1].to(self.device)
                mpid, ij, systems = data[2:]
                ng = g.batch_size

                gt_label = ij_labels(ij, systems, 'torch').to(self.device)
                ij_index = ij_labels(ij, ['triclinic'] * ng, 'torch').to(self.device)
                assert gt_label.shape == ij_index.shape

                pred, label = self.model(g, ij_index, gt_label, teacher_forcing)
                loss_reg = self.criterion_reg(pred, y)
                loss_class = self.criterion_class(label, gt_label)
                loss = loss_reg + self.weigth_class_loss * loss_class
                avgloss += np.array([loss_reg.detach().item(), loss_class.detach().item(), loss.detach().item()]) * ng
                acc = self.get_acc(label, gt_label, ng)
                acc_tot += acc * ng

                if i%10 == 0:
                    print(f"...[{epoch}|{i}|{mode}]: reg {loss_reg:.4f}, ml {loss_class:.4f}, tot {loss:.4f}, acc0 {acc[0]:.4f}, acc1 {acc[1]:.4f} acc {acc[2]:.4f}")

            avgloss /= eval_size
            acc_tot /= eval_size
        return avgloss, acc_tot

    def train(self, df_train, df_val, filename):
        tic = time.perf_counter()
        best_val_reg_loss, best_epoch = 1e6, 0

        loaders, sizes, stats = get_loaders(df_train, df_val, self.max_edge, self.bs, self.graph_params)
        train_loader, val_loader = loaders
        train_size, val_size = sizes
        mean, std = stats

        print('BEGIN TRAINING')
        print(f'save file: {filename}')

        for epoch in range(self.num_epochs):
            tic_epoch = time.perf_counter()

            train_loss, train_acc = self.train_step(epoch, train_loader, train_size)
            val_loss, val_acc = self.eval_step(epoch, val_loader, val_size, 'val')

            if val_loss[0] <= best_val_reg_loss:
                best_val_reg_loss = val_loss[0]
                best_epoch = epoch
                save_pt = 'save_params/params_' + filename + '_best.pt'
                torch.save(self.model.state_dict(), save_pt)

            if epoch % 10 == 0:
                save_pt = 'save_params/params_' + filename + '_' + str(epoch) + '.pt'
                torch.save(self.model.state_dict(), save_pt)

            toc_epoch = time.perf_counter()
            print(f"Current epoch: {epoch}")
            print(f"[train|{epoch}]: reg {train_loss[0]:.4f}, ml {train_loss[1]:.4f}, tot {train_loss[2]:.4f}, acc0 {train_acc[0]:.4f}, acc1 {train_acc[1]:.4f}, acc {train_acc[2]:.4f}")
            print(f"[val|{epoch}]: reg {val_loss[0]:.4f}, ml {val_loss[1]:.4f}, tot {val_loss[2]:.4f}, acc0 {val_acc[0]:.4f}, acc1 {val_acc[1]:.4f}, acc {val_acc[2]:.4f}")
            #print(f"Best val reg loss at epoch: {best_epoch}")
            #print(f"All losses at best val reg loss: {best_val_loss:.4f}")
            print(f'{self.device} run time [{epoch}]: {toc_epoch - tic_epoch} s \n')

        toc = time.perf_counter()
        print(f'total {self.device} run time: {toc - tic} s')

