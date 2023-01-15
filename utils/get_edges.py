from pymatgen.core import Structure, Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN

import numpy as np
import torch
import networkx as nx

class Edges:
    
    def __init__(self, struct, coord_type='cart', data_type='torch', max_edges=2000):
        assert isinstance(struct, Structure)
        self.struct = struct
        self.data_type = data_type
        self.coords = np.array([site.coords for site in struct.sites])
        self.frac_coords = np.array([site.frac_coords for site in struct.sites])
        self.coord_type = coord_type
        self.edge_style = None
        self.src = np.array([-1])
        self.max_edges = max_edges
        #self.edge_fn = getattr(self, edge_style)
        #self.edge_fn()
    
    
    def __repr__(self):
        return 'edge style: %s \nnumber of edges: %s \nnumber of nodes: %s' % \
                (self.edge_style, len(self.src), self.src.max().item() + 1)

        
    def _tensor(self, src, dst, dist, dr):
        if self.data_type == 'torch':
            self.src = torch.tensor(src, dtype=torch.long)
            self.dst = torch.tensor(dst, dtype=torch.long)
            self.dist = torch.tensor(dist, dtype=torch.float).view(-1, 1)
            self.dr = torch.tensor(dr, dtype=torch.float)
            
    
    def _elim_duplicate(self, edges):
        reduce = set(edges)
        edges = list(reduce)
        return edges


    def _check_scc(self, edges, mode):
        netx = nx.Graph()
        netx.add_nodes_from([i for i, _ in enumerate(self.coords)])
        netx.add_edges_from([edge for edge in edges])
        netx = netx.to_directed()
        scc = list(nx.strongly_connected_components(netx))
        if len(scc) > 1:
            if mode == 'bool':
                return True
            elif mode == 'info':
                return scc
        else:
            return False
    
    def adjacency_matrix(self):
        n = len(self.struct.sites)
        adj = np.zeros((n, n))
        for i, j in zip(self.src.tolist(), self.dst.tolist()):
            adj[i, j] += 1.
        return adj
    
    
    def degree_matrix(self):
        adj = self.adjacency_matrix()
        Di = np.sum(adj, axis=-1)
        Dj = np.sum(adj, axis=0)
        return Di, Dj
    
            
    def cell_radius(self, max_radius=12., extension=0.5, max_edges=10000, directional=True, primitive=False):
        if primitive:
            lat = SpacegroupAnalyzer(self.struct).get_primitive_standard_structure().lattice.matrix
        else:
            lat = self.struct.lattice.matrix
            
        radii = np.abs(lat).sum(axis=0) + extension
        for x, radius in enumerate(radii):
            if radius > max_radius:
                radii[x] = max_radius
        
        nbrs = self.struct.get_neighbor_list(radii.max())
        nbrs_all = self.struct.get_all_neighbors(radii.max())
        src, dst, dist, cell = nbrs[1], nbrs[0], nbrs[3], nbrs[2]
        #r_nbrs = [list(map(lambda r: (r.x, r.y, r.z), list(map(lambda x: x[0], nbr)))) for nbr in nbrs_all]
        #r_nbrs = [pos for r_nbr in r_nbrs for pos in r_nbr] 
        cart_nbrs = [tuple(r.coords) for nbr in nbrs_all for r in nbr]
        frac_nbrs = [tuple(r.frac_coords) for nbr in nbrs_all for r in nbr]
        ijdrc = [(i, j, d, 
                  tuple(np.array(r) - self.coords[i]), 
                  tuple(np.array(f) - self.frac_coords[i]), 
                  c.tolist()) 
                  for i, j, d, r, f, c in zip(src, dst, dist, cart_nbrs, frac_nbrs, cell)]

        if directional:
            ijdrc = np.array([(i, j, d, r, f, c) for i, j, d, r, f, c in ijdrc 
                             if (np.abs(r[0]) <= radii[0]) and (np.abs(r[1]) <= radii[1]) and (np.abs(r[2]) <= radii[2])], 
                             dtype=[('src', int), ('dst', int), ('dist', float), ('pos', tuple), ('frac', tuple), ('cell', list)])
        else:
            ijdrc = np.array([(i, j, d, r, f, c) for i, j, d, r, f, c in ijdrc], 
                             dtype=[('src', int), ('dst', int), ('dist', float), ('pos', tuple), ('frac', tuple), ('cell', list)])  

        ijdrc.sort(order='dist')
        ijdrc.sort(order='src')
        src = [i for i, _, _, _, _, _ in ijdrc] #+ [i for _, i, _, _, _, _ in ijdrc]
        dst = [j for _, j, _, _, _, _ in ijdrc] #+ [j for j, _, _, _, _, _ in ijdrc]
        dist = [d for _, _, d, _, _, _ in ijdrc] #+ [d for _, _, d, _, _, _ in ijdrc]
        if self.coord_type == 'cart':
            dr = [r for _, _, _, r, _, _ in ijdrc] # cart coords of dst
        elif self.coord_type == 'frac':
            dr = [f for _, _, _, _, f, _ in ijdrc] #+ [tuple([-1*i for i in f]) for _, _, _, _, f, _ in ijdrc]
        cell = [c for _, _, _, _, _, c in ijdrc] #+ [c for _, _, _, _, _, c in ijdrc]
        #self._tensor(src, dst, dist, np.array(pos) - self.coords[src])
        self._tensor(src, dst, dist, dr)
        self.cell = torch.tensor(cell, dtype=torch.float)
        self.edge_style = 'cell_radius'
        '''
        if len(ijdr) > max_edges:
            #self.from_tfn(radii.min())
            self.voronoinn()
        '''
        return self

    
    def structuregraph(self, strategy, *args, **kwargs):
        G = StructureGraph.with_local_env_strategy(self.struct, strategy)
        adj = G.as_dict()['graphs']['adjacency']
        src, dst, dist, pos, pos_src, pos_dst, cell = [], [], [], [], [], [], []
        
        for idx, vals in enumerate(adj):
            src += [idx] * len(vals)
            dst += [val['id'] for val in vals]
            dist += [self.struct.sites[idx].distance(self.struct.sites[val['id']], jimage=val['to_jimage']) for val in vals]
            if self.coord_type == 'cart': 
                pos_src += [self.struct.sites[idx].coords] * len(vals)
                pos_dst += [self.struct.sites[val['id']].coords + 
                            np.einsum('ab,a->b', self.struct.lattice.matrix, np.array(val['to_jimage'])) for val in vals]
            elif self.coord_type == 'frac':
                pos_src += [self.struct.sites[idx].frac_coords] * len(vals)
                pos_dst += [self.struct.sites[val['id']].frac_coords + val['to_jimage'] for val in vals]
            cell += [val['to_jimage'] for val in vals]

        # for undirected graph
        src0 = src.copy()
        src += dst
        dst += src0
        dist += dist
        pos += pos
        dr = [r2 - r1 for r1, r2 in zip(pos_src, pos_dst)] + [r2 - r1 for r1, r2 in zip(pos_dst, pos_src)]

        assert len(src) == len(dst) == len(dist) == len(dr)
        self._tensor(src, dst, dist, dr)
        self.cell = torch.tensor(cell + [(-1*h, -1*k, -1*l) for h, k, l in cell]).float()
        self.edge_style = str(strategy)
        return self 
    
    def crystalnn(self, distance_cutoffs=[0.5, 1.0], *args, **kwargs):
        return self.structuregraph(CrystalNN(distance_cutoffs, *args, **kwargs))
        
    def voronoinn(self, cutoff=13.0, *args, **kwargs):
        return self.structuregraph(VoronoiNN(cutoff=cutoff, **kwargs))
    
    
    # Copied from TFN code
    def from_tfn(self, radius=3., self_interaction=False, r_min=1e-8):
        """
        Create neighbor list (edge_index) and relative vectors (edge_attr)
        based on radial cutoff and periodic lattice.

        :param pos: torch.tensor of coordinates with shape (N, 3)
        :param r_max: float of radial cutoff
        :param self_interaction: whether or not to include self edge

        :return: list of edges [(2, num_edges)], Tensor of relative vectors [num_edges, 3]

        edges are given by the convention
        edge_list[0] = source (convolution center)
        edge_list[1] = target (neighbor index)

        Thus, the edge_list has the same convention vector notation for relative vectors
        \vec{r}_{source, target}

        Relative vectors are given for the different images of the neighbor atom within r_max.
        """
        # To preserve the old code, we add our parameters here.
        pos = torch.tensor([site.coords for site in self.struct.sites])
        lattice = self.struct.lattice.matrix
        r_max = radius
        ###
        
        N, _ = pos.shape
        structure = Structure(lattice, ['H'] * N, pos, coords_are_cartesian=True)

        nei_list = []
        geo_list = []

        neighbors = structure.get_all_neighbors(
            r_max,
            include_index=True,
            include_image=True,
            numerical_tol=r_min
        )
        for i, (site, neis) in enumerate(zip(structure, neighbors)):
            indices, cart = zip(*[(n.index, n.coords) for n in neis])
            cart = torch.tensor(cart)
            indices = torch.LongTensor([[i, target] for target in indices])
            dist = cart - torch.tensor(site.coords)
            if self_interaction:
                self_index = torch.LongTensor([[i, i]])
                indices = torch.cat([self_index, indices], dim=0)
                self_dist = pos.new_zeros(1, 3, dtype=dist.dtype)
                dist = torch.cat([self_dist, dist], dim=0)
            nei_list.append(indices)
            geo_list.append(dist)
        
        # from this code, dist is dr -> pos in our code and edge_attr is dist instead.
        edge_idx, edge_attr = torch.cat(nei_list, dim=0).transpose(1, 0), torch.cat(geo_list, dim=0)
        dist = torch.sum(torch.pow(edge_attr, 2), dim=-1).view(-1, 1)
        #self.edge_index_dict, self.edge_edges, self.edge_edge_index = _get_edge_edges_and_index(edge_idx, symmetric_edges=False)       
        self._tensor(edge_idx[0], edge_idx[1], dist, edge_attr)
        self.edge_style = 'from_tfn'
        return self

