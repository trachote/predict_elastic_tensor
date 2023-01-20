import sys

import numpy as np
import torch

from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules_deep import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, 
                 atom_feat_size: int,
                 num_channels: int, 
                 num_nlayers: int=1,
                 num_degrees: int=4, 
                 edge_dim: int=4, 
                 div: float=4, 
                 pooling: str='avg', 
                 n_heads: int=4, 
                 radial_dim: int=32, 
                 embed_dim=None, 
                 **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.radial_dim = radial_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        if embed_dim is not None:
            self.embedding = nn.Embedding(95, embed_dim)

        self.fibers = {'in': Fiber(1, atom_feat_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        self.Gblock, self.Greg, self.Gclass = self._build_gcn(self.fibers, 1)

        # Pooling
        if self.pooling == 'avg':
            self.pool = GAvgPooling()
        elif self.pooling == 'max':
            self.pool = GMaxPooling()

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock, Greg, Gclass = [], [], []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads, mid_dim=self.radial_dim))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        
        last_attention = False 
        if last_attention:
            Greg.append(GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads, mid_dim=self.radial_dim))
            Gclass.append(GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads, mid_dim=self.radial_dim))
        else:
            Greg.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim,
                                   mid_dim=self.radial_dim))
            Gclass.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim,
                                   mid_dim=self.radial_dim))


        return nn.ModuleList(Gblock), nn.ModuleList(Greg), nn.ModuleList(Gclass)

    def forward(self, G):
        # Embedding atomic features
        feats = [G.ndata['f']]
        if self.embed_dim is not None:
            feats.append(self.embedding(G.ndata['z']).unsqueeze(-1))
        G.ndata['f'] = torch.cat(feats, dim=-2)

        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)
        
        hr = h
        for layer in self.Greg:
            hr = layer(hr, G=G, r=r, basis=basis)

        hc = h
        for layer in self.Gclass:
            hc = layer(hc, G=G, r=r, basis=basis)

        hr = self.pool(hr, G=G)
        hc = self.pool(hc, G=G)
        return hr, hc

class MLPClass(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 num_degrees: int=4, 
                 **kwargs):
        super().__init__()
        # Build the network
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        
        self.dim = 128
        self.ijemb = nn.Sequential(nn.Linear(21, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        self.fibers = {'out': Fiber(1, num_degrees*self.num_channels)}
        self.mlp = self._build_mlp(self.fibers, 21)
        
    def _build_mlp(self, fibers, out_dim):
        MLblock = []
        MLblock.append(nn.Linear(fibers['out'].n_features+self.dim, fibers['out'].n_features+self.dim))
        MLblock.append(nn.ReLU(inplace=True))
        MLblock.append(nn.Linear(fibers['out'].n_features+self.dim, out_dim))
        #MLblock.append(nn.Softmax(dim=-1))
        #MLblock.append(nn.Sigmoid())
        return nn.ModuleList(MLblock)

    def forward(self, x, ij_index):
        ij = self.ijemb(ij_index)
        x = torch.cat([x, ij], dim=-1)
        for layer in self.mlp:
            x = layer(x)
        return x

class MLPReg(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 num_degrees: int=4, 
                 **kwargs):
        super().__init__()
        # Build the network
        self.num_channels = num_channels
        self.num_degrees = num_degrees

        self.fibers = {'out': Fiber(1, num_degrees*self.num_channels)}
        self.emb, self.mlp = self._build_mlp(self.fibers, 1)

    def _build_mlp(self, fibers, out_dim):
        # Embed ij
        dim = 128
        IJblock = nn.Sequential(nn.Linear(21, dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(dim, dim))
    
        # FC layers
        MLblock = []
        MLblock.append(nn.Linear(self.fibers['out'].n_features+dim, self.fibers['out'].n_features+dim))
        MLblock.append(nn.ReLU(inplace=True))
        MLblock.append(nn.Linear(self.fibers['out'].n_features+dim, out_dim))
        return IJblock, nn.ModuleList(MLblock)

    def forward(self, h, x):
        x = self.emb(x)
        h = torch.cat((h, x), dim=-1)
        for layer in self.mlp:
            h = layer(h)
        return h

class SE3Net(nn.Module):
    def __init__(self, num_layers: int, 
                 atom_feat_size: int, 
                 num_channels: int, 
                 num_nlayers: int=1,
                 num_degrees: int=4, 
                 edge_dim: int=4, 
                 div: float=4, 
                 pooling: str='avg', 
                 n_heads: int=4, 
                 radial_dim: int=32, 
                 embed_dim=None,
                 **kwargs):
        super().__init__()
        atom_feat_size += embed_dim if embed_dim is not None else 0
        self.gnn = SE3Transformer(num_layers, atom_feat_size, num_channels, num_nlayers=num_nlayers, 
                           num_degrees=num_degrees, edge_dim=edge_dim, div=div, pooling=pooling, 
                           n_heads=n_heads, embed_dim=embed_dim, radial_dim=radial_dim)
        self.mlp_reg = MLPReg(num_channels, num_degrees)
        self.mlp_class = MLPClass(num_channels, num_degrees)
    
    def forward(self, G, ij_index, gt_label=None, teacher_forcing=False):
        hr, hc = self.gnn(G)
        y = self.mlp_class(hc, ij_index)
        if teacher_forcing:
            x = gt_label
        else:
            x = torch.sigmoid(y)
        pred = self.mlp_reg(hr, x)
        return pred, torch.sigmoid(y)
