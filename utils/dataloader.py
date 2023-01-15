import dgl
#import pandas as pd
from utils.etc import ijdict_symmetric
import torch
from torch.utils.data import DataLoader, ConcatDataset

CRYSTAL_SYSTEMS = ['cubic', 'tetragonal', 
                   'hexagonal', 'trigonal', 
                   'orthorhombic', 'monoclinic', 
                   'triclinic']

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y, dtype=torch.float), \
           [g.mpid for g in graphs], [g.ij for g in graphs]

def get_loaders(df_train, df_val, max_edge, bs, graph_params, shuffle=True):     
    train_set, val_set = [], []
    mean, std = 0., 0.
    
    for system in CRYSTAL_SYSTEMS:
        df1 = df_train[(df_train['spacegroup.crystal_system'] == system) & (df_train['edge_size'] <= max_edge)]
        df2 = df_val[(df_val['spacegroup.crystal_system'] == system) & (df_val['edge_size'] <= max_edge)]
        print(f'crystal system: {system}, train/val: {len(df1)}/{len(df2)}')
       
        train_s = MPtoGraph(df1, graph_params.train)
        print('\ttrain pass') 
         
        val_s = MPtoGraph(df2, graph_params.val)
        print('\tval pass') 

        train_s.set_mode(ijdict_symmetric('train')[system])
        val_s.set_mode(ijdict_symmetric('val')[system])
   
        mean += train_s.mean * len(train_s)
        std += train_s.std * len(train_s)
        train_set.append(train_s), val_set.append(val_s)
   
    train_size, val_size = sum([len(n) for n in train_set]), sum([len(n) for n in val_set])
    mean = mean / train_size 
    std = std / train_size 
    for a, b in zip(train_set, val_set):
        a.mean, a.std = mean, std
        b.mean, b.std = mean, std

    train_set = ConcatDataset(train_set)
    val_set = ConcatDataset(val_set)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=shuffle, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=2*bs, shuffle=False, collate_fn=collate)
    
    return (train_loader, val_loader),\
           (train_size, val_size),\
           (mean, std)
