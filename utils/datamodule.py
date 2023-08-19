import dgl
#import pandas as pd
from utils.etc import ijdict_symmetric
from utils.get_strained_crystal_new import MPtoGraph
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

CRYSTAL_SYSTEMS = ['cubic', 'tetragonal', 
                   'hexagonal', 'trigonal', 
                   'orthorhombic', 'monoclinic', 
                   'triclinic']

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y, dtype=torch.float), \
           [g.mat_id for g in graphs], [g.ij for g in graphs], \
           [g.system for g in graphs]

def split_dataset(df, val_ratio, edge_style, seed=0):
    df = df[df[edge_style + '_connected_graph']]
    bins = np.linspace(0, 1500, 1)
    y_binned = np.digitize(df['elasticity.K_Voigt'], bins)
    train_df, val_df, _, _ = train_test_split(df, df, 
                                              test_size=val_ratio, 
                                              stratify=y_binned, 
                                              random_state=seed)
    return train_df, val_df

def get_loaders(train_df, val_df, max_edge, bs, graph_params, shuffle=True):     
    train_set, val_set = [], []
    mean, std = 0., 0.
    
    for system in CRYSTAL_SYSTEMS:
        df1 = train_df[(train_df['spacegroup.crystal_system'] == system) & (train_df['edge_size'] <= max_edge)]
        df2 = val_df[(val_df['spacegroup.crystal_system'] == system) & (val_df['edge_size'] <= max_edge)]
        print(f'crystal system: {system.upper()}')  
        print(f'\tnumber of train : {len(df1)}') 
        print(f'\tnumber of val: {len(df2)}\n') 
       
        if len(df1) > 0:
            train_s = MPtoGraph(df1, **graph_params.train)
            train_s.set_mode(ijdict_symmetric('train')[system])
            train_set.append(train_s)
        
        if len(df2) > 0:
            val_s = MPtoGraph(df2, **graph_params.val)
            val_s.set_mode(ijdict_symmetric('val')[system])
            val_set.append(val_s)

        mean += train_s.mean * len(train_s)
        std += train_s.std * len(train_s)
        
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

def get_eval_loader(df, bs, graph_params):
    dataset = MPtoGraph(df, **graph_params.val, task='eval')
    return DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collate)
