from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import dgl
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R3
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.etc import *
from utils.get_edges import Edges


class MPtoGraph(Dataset):
    def __init__(self, df, target,
                 shuffle=False,
                 strain=2,
                 weight='ones',
                 one_hot_size=95,
                 atom_feats=None,
                 conventional_cell=True,
                 recenter=True,
                 frac_coord=False,
                 edge_style='cell_radius',
                 graph_object='dgl',
                 train_energy=True,
                 normalize_target=True,
                 rotations=None,
                 task='train',
                 **kwargs
                ):

        self.df = df
        self.target = target
        self.weight = weight
        self.strain = strain
        self.eij = get_epsilon(strain)
        self.e_vector = strain_vector(strain)
        self.con_cell = conventional_cell
        self.one_hot_size = one_hot_size
        self.recenter = recenter
        self.frac_coord = frac_coord
        self.edge_style = edge_style
        self.graph_object = graph_object
        self.train_energy = train_energy
        self.atom_feats = atom_feats
        self.normalize_target = normalize_target
        self.rotations = rotations
        self.task = task
        self.dir_path = os.path.dirname(os.path.realpath(__file__))        

        if shuffle:
            self.df = self.df.sample(frac=1)

        self.ijkey = {i*6-sum([k for k in range(i+1)])+j: (i, j) 
                      for i in range(6) for j in range(i, 6)} # {0: (0,0), 1: (0,1), ..., 20: (5,5)}

        if self.task == 'train':
            self.load_data()
        self.data_len = len(self.df)
        self.multiply = len(self.ijkey)
        self.len = self.data_len * self.multiply

    def __len__(self):
        return self.len

    def load_data(self):
        self.all_y = np.array(self.df['elasticity.elastic_tensor'].tolist())
        self.unit = 'GPa'

        if self.train_energy:
            self._convert_to_energy()

        self.flatten_y = self.all_y.reshape(-1, 36)
        norm_ls = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 28, 29, 35]
        self.mean = np.mean(self.flatten_y[:, norm_ls])
        self.std = np.std(self.flatten_y[:, norm_ls])

    def _convert_to_energy(self):
        unit_converter = 1000./160.2176487 # convert to meV
        volume, nsites = self.df['volume'].values, self.df['nsites'].values
        const = 0.5 * volume * unit_converter / nsites / 10 ** 4
        dU = np.einsum('abi,nij,abj->nab', self.e_vector, self.all_y, self.e_vector) * const.reshape(-1, 1, 1)
        self.all_y = dU
        self.unit = 'meV'

    def set_mode(self, ijkey: dict={}):
        self.ijkey = ijkey
        self.multiply = len(self.ijkey)
        self.load_data()
        self.len = self.data_len * self.multiply

    def get_target(self, idx):
        y = self.all_y[idx]
        if self.normalize_target:
            y = (y - self.mean) / self.std
        return y

    def _to_one_hot(self, z, max_z):
        one_hots = []
        for zi in z:
            one_hot = torch.zeros((max_z, 1), dtype=torch.float)
            one_hot[zi-1, :] = 1
            one_hots.append(one_hot.tolist())
        return torch.tensor(one_hots, dtype=torch.float)

    def _weight_fn(self, w, **kwargs):
        if self.weight == 'ones':
            return torch.tensor([[1]] * len(w), dtype=torch.float)
        elif self.weight == 'distance':
            return w
        elif self.weight == 'distance_square':
            return torch.pow(w, 2)
        elif self.weight == 'distance_pow4':
            return torch.pow(w, 4) / 10.
        elif self.weight == 'inverse':
            return 10. / w
        elif self.weight == 'inverse_square':
            return 100. / torch.pow(w, 2)

    def _frac_coords(self, matrix, v):
        norm = np.linalg.norm(matrix, axis=-1)
        assert norm.shape[-1] == v.shape[-1]
        return torch.tensor(v / norm, dtype=torch.float)

    def _recenter(self, coords):
        center = coords.mean(dim=0)
        assert coords.shape[-1] == center.shape[-1]
        return coords - center

    def get_atom_feats(self, elems):
        feat = []
        feat += [self._to_one_hot([elem.number for elem in elems], self.one_hot_size)] if 'Z' in self.atom_feats else []
        feat += [torch.tensor([[[elem.atomic_mass]] for elem in elems])] if 'mass' in self.atom_feats else []
        feat += [torch.tensor([[[elem.X]] for elem in elems])] if 'X' in self.atom_feats else []
        feat += [torch.tensor([[[elem.atomic_radius if elem.atomic_radius != None else elem.atomic_radius_calculated]] for elem in elems])] if 'radius' in self.atom_feats else []
        if 'polar' in self.atom_feats:
            """
            P. Schwerdtfeger and J. K. Nagle, 2018 
            table of static dipole polarizabilities of 
            the neutral elements in the pe-riodic table, 
            Molecular Physics 117, 1200 (2019)
            """
            polar = pd.read_csv(os.path.join(self.dir_path, 'elemental_polarizability.csv'))
            feat.append(torch.tensor([[[polar[polar['Z'] == elem.number]['Polarizability'].item()]] for elem in elems]))
        feat = torch.cat(feat, dim=1).float()
        return feat

    def _graph_elements(self, struct, lat):
        if self.frac_coord:
            coords = [site.frac_coords for site in struct.sites]
        else:
            coords = [item.coords for item in struct.sites]
        coords = torch.tensor(coords, dtype=torch.float)

        edges = getattr(Edges(struct, coord_type='frac'), self.edge_style)(max_radius=12., extension=0.5)
        src, dst, dist, dr, T = edges.src, edges.dst, edges.dist, edges.dr, edges.cell
        dr = torch.einsum('ni,ij->nj', dr, lat)
        elems = [struct.species[n] for n in range(len(struct.sites))]

        if self.rotations is not None:
            if self.rotations == 'random':
                angles = np.random.randint(0, 359, size=3)
            else:
                angles = self.rotations
            R = R3.from_euler('zyx', angles, degrees=True).as_matrix()
            dr = torch.einsum('ij,nj->ni', torch.from_numpy(R).float(), dr)

        if self.frac_coord:
            dr = self._frac_coords(struct.lattice.matrix, dr)
        if self.recenter:
            coords = self._recenter(coords)

        feat = self.get_atom_feats(elems)
        z = torch.tensor([elem.Z for elem in elems], dtype=torch.long)
        edge_attr = self._weight_fn(dist, sites=elems, src=src, dst=dst).float()
        cell = torch.einsum('ij,ni->nj', torch.tensor(struct.lattice.matrix.tolist()).float(), edges.cell)
        return coords, feat, z, src, dst, dr, edge_attr, cell

    def to_dgl(self, struct, lat):
        pos, feat, z, src, dst, dr, w, T = self._graph_elements(struct, lat)
        G = dgl.graph((src, dst))
        G.ndata['x'] = pos
        G.ndata['f'] = feat
        G.ndata['z'] = z
        G.edata['d'] = dr
        G.edata['w'] = w
        return G

    def to_pyg(self, struct, lat):
        pos, x, z, src, dst, dr, w, T = self._graph_elements(struct, lat)
        edge_index = torch.cat((src.reshape(1, -1), dst.reshape(1, -1)))
        return Data(x=x.squeeze(-1), pos=pos, edge_index=edge_index, edge_attr=dr, z=z, T=T)

    def _get_ij(self, data_idx, idx):
        n = int(idx / (self.data_len))
        ijlist = list(self.ijkey.values())
        return ijlist[n]

    def strained(self, struct, eij):
        ## The equation from [S Q Wang and H Q Ye 2003 J. Phys.: Condens. Matter 15 5307] was incorrect.
        #strained_lat = np.matmul(struct.lattice.matrix, eij)
        strained_lat = np.matmul(eij, struct.lattice.matrix)
        strained_str = struct.copy()
        strained_str.lattice = Lattice(strained_lat)
        return strained_str, torch.from_numpy(strained_lat).float()

    def __getitem__(self, idx):
        data_idx = idx % self.data_len
        struct = Structure.from_str(self.df['cif'][data_idx], 'cif')
        if self.con_cell:
            struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
            ## an a axis of the hexagonal lattice will be aligned on x axis by using get_refined_structure.
            # struct = SpacegroupAnalyzer(struct).get_refined_structure()

        tensor_idx = self._get_ij(data_idx, idx)
        self.ij_idx = tensor_idx
        if tensor_idx == (7, 7):
            eij = torch.eye(3)
        else:
            eij = self.eij[tensor_idx[0], tensor_idx[1]]
        strained_str, strained_lat = self.strained(struct, eij)
        pg = SpacegroupAnalyzer(strained_str).get_point_group_symbol()
        system = SpacegroupAnalyzer(strained_str).get_crystal_system()

        if self.graph_object == 'dgl':
            Gs = self.to_dgl(struct, strained_lat)
        elif self.graph_object == 'pyg':
            Gs = self.to_pyg(struct, strained_lat)
        Gs.ij, Gs.system = tensor_idx, system
        Gs.mat_id = 'material_' + f'{data_idx:03d}'

        if self.task == 'train':
            y = self.get_target(data_idx)
            ## cij and lattice parameters are not aligned on the same axes; therefore, data must be transformed.
            y = data_alignment(struct, y)
            if tensor_idx == (7, 7):
                y = - self.mean / self.std
            else:
                y = y[tensor_idx[0]][tensor_idx[1]]
            mat_id = self.df.index[data_idx]
            Gs.mat_id, Gs.pg = mat_id, pg
        elif self.task == 'eval':
            y = 0.

        if self.graph_object == 'pyg':
            Gs.y = torch.tensor(y, dtype=torch.float).view(-1, 1)
            Gs.lattice = strained_lat
            return Gs
        elif self.graph_object == 'dgl':
            return (Gs, [y])
        
