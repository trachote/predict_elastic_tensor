import numpy as np
import torch
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def strain_vector(strain):
    vector = np.zeros((6, 6, 6))
    for i in range(6):
        for j in range(6):
            vector[i, j, i] = strain
            vector[i, j, j] = strain
    return vector


def get_epsilon(strain):
    e1, e4 = 1 + strain/100, strain/200
    I = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    exy = np.zeros((6, 6, 3, 3))

    def idx(i):
        if i == 3:
            return 1, 2
        if i == 4:
            return 0, 2
        if i == 5:
            return 0, 1

    for i in range(6):
        for j in range(6):
            ep = I.copy()
            if i < 3:
                ep[i, i] = e1
                if j < 3:
                    ep[j, j] = e1
                elif j == 3:
                    ep[1, 2] = ep[2, 1] = e4
                elif j == 4:
                    ep[0, 2] = ep[2, 0] = e4
                elif j == 5:
                    ep[0, 1] = ep[1, 0] = e4 
                exy[i, j] = ep
            elif i >= 3:
                if j < 3:
                    exy[i, j] = exy[j, i]
                elif i == j:
                    l, m = idx(i)
                    ep[l, m] = ep[m, l] = e4
                    exy[i, j] = ep
                else:
                    li, mi = idx(i)
                    lj, mj = idx(j)
                    ep[li, mi] = ep[mi, li] = ep[lj, mj] = ep[mj, lj] = e4
                    exy[i, j] = ep
    return exy


def data_alignment(struct, y):
        lat = struct.lattice.matrix
        bravias = SpacegroupAnalyzer(struct).get_crystal_system()
        if bravias == 'tetragonal':
            if y[0][0] == y[1][1]:
                ls = [0, 1, 2, 3, 4, 5]
            elif y[0][0] == y[2][2]:
                ls = [0, 2, 1, 3, 5, 4]
            elif y[1][1] == y[2][2]:
                ls = [2, 1, 0, 5, 4, 3]
            else:
                ls = [0, 1, 2, 3, 4, 5]
            ls = np.meshgrid(ls, ls)
            return y[ls[0], ls[1]]
        
        elif bravias == 'orthorhombic':
            lat = [lat[0][0], lat[1][1], lat[2][2]]
            b = lat.index(max(lat))
            c = lat.index(min(lat))
            a = [0, 1, 2]
            if b < c:
                a.pop(b)
                a.pop(c-1)
            elif c < b:
                a.pop(c)
                a.pop(b-1)
            sw = {a[0]: 0 , b: 1, c: 2}
            ls = [sw[0], sw[1], sw[2], a[0]+3, b+3, c+3]
            ls = np.meshgrid(ls, ls)
            return y[ls[0], ls[1]]
        
        else:
            return y
        
        
def pg_vec_from_sg(space_group_number, vector=True):
    vec = np.zeros(32)
    if space_group_number <= 2: # 1, -1
        vec[space_group_number - 1] = 1.
    elif space_group_number <= 5: # 2
        vec[2] = 1.
    elif space_group_number <= 9: # m
        vec[3] = 1.
    elif space_group_number <= 15: # 2/m
        vec[4] = 1.
    elif space_group_number <= 24: # 222
        vec[5] = 1.
    elif space_group_number <= 46: # mm2
        vec[6] = 1.
    elif space_group_number <= 74: # 2/m 2/m 2/m
        vec[7] = 1.
    elif space_group_number <= 80: # 4
        vec[8] = 1.
    elif space_group_number <= 82: # -4
        vec[9] = 1.
    elif space_group_number <= 88: # 4/m
        vec[10] = 1.
    elif space_group_number <= 98: # 422
        vec[11] = 1.
    elif space_group_number <= 110: # 4mm
        vec[12] = 1.
    elif space_group_number <= 122: # -42m
        vec[13] = 1.
    elif space_group_number <= 142: # 4/m 2/m 2/m
        vec[14] = 1.
    elif space_group_number <= 146: # 3
        vec[15] = 1.
    elif space_group_number <= 148: # -3
        vec[16] = 1.
    elif space_group_number <= 155: # 32
        vec[17] = 1.
    elif space_group_number <= 161: # 3m
        vec[18] = 1.
    elif space_group_number <= 167: # -3 2/m
        vec[19] = 1.
    elif space_group_number <= 173: # 6
        vec[20] = 1.
    elif space_group_number == 174: # -6
        vec[21] = 1.
    elif space_group_number <= 176: # 6/m
        vec[22] = 1.
    elif space_group_number <= 182: # 622
        vec[23] = 1.
    elif space_group_number <= 186: # 6mm
        vec[24] = 1.
    elif space_group_number <= 190: # -6m2
        vec[25] = 1.
    elif space_group_number <= 194: # 6/m 2/m 2/m
        vec[26] = 1.
    elif space_group_number <= 199: # 23
        vec[27] = 1.
    elif space_group_number <= 206: # 2/m -3
        vec[28] = 1.
    elif space_group_number <= 214: # 432
        vec[29] = 1.
    elif space_group_number <= 220: # -43m
        vec[30] = 1.
    elif space_group_number <= 230: # 4/m -3 2/m
        vec[31] = 1.
    
    if ~vector:
        vec = vec.tolist().index(1.)
    return vec


def ijdict_crystal_systems(mode='train'):
    all_ij = [(i, j) for i in range(6) for j in range(i, 6)] 
    cub_ij = [(0, 0), (0, 1), (0, 3), (3, 3), (3, 4)]
    hex_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    tet_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    rho_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (2, 2), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    rho2_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    
    if mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_on_diag(mode='train'):
    all_ij = [(i, i) for i in range(6)]
    cub_ij = [(0, 0), (3, 3)]
    hex_ij = [(0, 0), (2, 2), (3, 3), (5, 5)]
    tet_ij = [(0, 0), (2, 2), (3, 3), (5, 5)]
    rho_ij = [(0, 0), (2, 2), (3, 3), (5, 5)]
    rho2_ij = [(0, 0), (2, 2), (3, 3), (5, 5)]

    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_off_diag(mode='train'):
    all_ij = [(i, j) for i in range(6) for j in range(i, 6) if i != j]
    cub_ij = [(0, 1), (0, 3), (0, 4), (3, 4)]
    hex_ij = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 3), (2, 5), (3, 4), (3, 5)]
    tet_ij = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5)]
    rho_ij = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (2, 5), (3, 4), (3, 5), (4, 5)]
    rho2_ij = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]
    
    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_bulk(mode='train'):
    all_ij = [(i, j) for i in range(3) for j in range(i, 3)]
    cub_ij = [(0, 0), (0, 1)]
    hex_ij = [(0, 0), (0, 1), (0, 2), (2, 2)]
    tet_ij = [(0, 0), (0, 1), (0, 2), (2, 2)]
    rho_ij = [(0, 0), (0, 1), (0, 2), (2, 2)]
    rho2_ij = [(0, 0), (0, 1), (0, 2), (2, 2)]
    
    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_shear(mode='train'):
    all_ij = [(i, j) for i in range(3, 6) for j in range(i, 6)]
    cub_ij = [(3, 3), (3, 4)]
    hex_ij = [(3, 3), (3, 4), (3, 5), (5, 5)]
    tet_ij = [(3, 3), (3, 4), (3, 5), (5, 5)]
    rho_ij = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    rho2_ij = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    
    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_mix(mode='train'):
    all_ij = [(i, j) for i in range(3) for j in range(3, 6)]
    cub_ij = [(0, 3)]
    hex_ij = [(0, 3), (0, 5), (2, 3), (2, 5)]
    tet_ij = [(0, 3), (0, 5), (1, 5), (2, 3), (2, 5)]
    rho_ij = [(0, 3), (0, 4), (0, 5), (1, 3), (2, 5)]
    rho2_ij = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 3), (2, 5)]
    
    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_symmetric(mode='train', target='all'):
    cub_ij = [(0, 0), (0, 1), (0, 3), (0, 4), (3, 3), (3, 4)]
    hex_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    tet_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    rho_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (2, 2), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    rho2_ij = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    
    if target == 'all':        
        all_ij = [(i, j) for i in range(6) for j in range(i, 6)]
    elif target == 'on_diag':
        all_ij = [(i, i) for i in range(6)]
    elif target == 'off_diag':
        all_ij = [(i, j) for i in range(6) for j in range(i, 6) if i != j]
    elif target == 'bulk':
        all_ij = [(i, j) for i in range(3) for j in range(i, 3)]
    elif target == 'shear':
        all_ij = [(i, j) for i in range(3, 6) for j in range(i, 6)]
    elif target == 'mix':
        all_ij = [(i, j) for i in range(3) for j in range(3, 6)]
    
    cub_ij = [item for item in cub_ij if item in all_ij]
    hex_ij = [item for item in hex_ij if item in all_ij]
    tet_ij = [item for item in tet_ij if item in all_ij]
    rho_ij = [item for item in rho_ij if item in all_ij]
    rho2_ij = [item for item in rho2_ij if item in all_ij]
    
    if mode == 'val':
        cub_ij = hex_ij = tet_ij = rho_ij = rho2_ij = all_ij
    elif mode == 'test':
        cub_ij = [item for item in all_ij if item not in cub_ij]
        hex_ij = [item for item in all_ij if item not in hex_ij]
        tet_ij = [item for item in all_ij if item not in tet_ij]
        rho_ij = [item for item in all_ij if item not in rho_ij]
        rho2_ij = [item for item in all_ij if item not in rho2_ij]

    cubic = {k: ij for k, ij in enumerate(cub_ij)}
    hexa = {k: ij for k, ij in enumerate(hex_ij)}
    tetraII = {k: ij for k, ij in enumerate(tet_ij)}
    rhomI = {k: ij for k, ij in enumerate(rho_ij)}
    rhomII = {k: ij for k, ij in enumerate(rho2_ij)}
    everyij = {k: ij for k, ij in enumerate(all_ij)}

    return {'cubic': cubic,
            'hexagonal': hexa,
            'tetragonal': tetraII,
            'trigonal': rhomII,
            'orthorhombic': everyij,
            'monoclinic': everyij,
            'triclinic': everyij}


def ijdict_laue(mode='train', target='all'):
    ij_dict = {}
    ij_dict['cub1'] = [(0, 0), (0, 1), (0, 3), (0, 4), (3, 3), (3, 4)]
    ij_dict['cub2'] = [(0, 0), (0, 1), (0, 3), (0, 4), (3, 3), (3, 4), (3, 5)]
    ij_dict['hex1'] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    ij_dict['tet1'] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (5, 5)]
    ij_dict['tet2'] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    ij_dict['rho1'] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (2, 2), (2, 5), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    
    if target == 'all':        
        all_ij = [(i, j) for i in range(6) for j in range(i, 6)]
    elif target == 'on_diag':
        all_ij = [(i, i) for i in range(6)]
    elif target == 'off_diag':
        all_ij = [(i, j) for i in range(6) for j in range(i, 6) if i != j]
    elif target == 'bulk':
        all_ij = [(i, j) for i in range(3) for j in range(i, 3)]
    elif target == 'shear':
        all_ij = [(i, j) for i in range(3, 6) for j in range(i, 6)]
    elif target == 'mix':
        all_ij = [(i, j) for i in range(3) for j in range(3, 6)]
        
    for sys in ['cub1', 'cub2', 'hex1', 'hex2', 'tet1', 'tet2', 'rho1', 'rho2', 'orth', 'mono', 'tric']:
        if sys not in ['hex2', 'rho2', 'orth', 'mono', 'tric']:
            ij_dict[sys] = [item for item in ij_dict[sys] if item in all_ij]
            if mode == 'val':
                ij_dict[sys] = all_ij
            elif mode == 'test':
                ij_dict[sys] = [item for item in all_ij if item not in ij_dict[sys]]
            ij_dict[sys] = {k: ij for k, ij in enumerate(ij_dict[sys])}
        
        else:
            ij_dict[sys] = {k: ij for k, ij in enumerate(all_ij)}

    return ij_dict 


def ij_labels(ij, bravais, dtype='numpy'):
    idx = [i*5 + j - sum(list(range(i))) for i, j in ij]
    sym = {'cubic': np.array([set([0, 6, 11]), set([1, 2, 7]), 
                              set([3, 9 ,14]), set([4, 5, 8, 10, 12, 13]), 
                              set([15, 18, 20]), set([16, 17, 19])]),
           
           'hexagonal': np.array([set([0, 6]), set([1]), set([2, 7]), 
                                  set([3, 9]), set([4, 8]), set([5, 10]), 
                                  set([11]), set([12, 13]), set([14]), 
                                  set([15, 18]), set([16]), set([17, 19]), 
                                  set([20])]),
           
           'tetragonal': np.array([set([0, 6]), set([1]), set([2, 7]), set([3, 9]), 
                                   set([4, 8]), set([5]), set([10]), set([11]), 
                                   set([12, 13]), set([14]), set([15, 18]), set([16]), 
                                   set([17, 19]), set([20])]),
           
           'trigonal': np.array([set([0, 6]), set([1]), set([2, 7]), set([3, 19]), 
                                 set([4]), set([5, 10]), set([8]), set([9, 17]), set([11]), 
                                 set([12, 13]), set([14]), set([15, 18]), set([16]), set([20])]),
           
           'orthorhombic': np.array([set([i]) for i in range(21)]),
           'monoclinic': np.array([set([i]) for i in range(21)]),
           'triclinic': np.array([set([i]) for i in range(21)])
          }
    masks = [list(sym[b][[s.intersection([i]) != set([]) for s in sym[b]]].item()) for i, b in zip(idx, bravais)]
    labels = np.zeros((len(ij), 21))
    for l, m in zip(labels, masks):
        l[m] = 1
    if dtype == 'numpy':
        return labels
    elif dtype == 'torch':
        return torch.tensor(labels).float()

    
def rand_ij_index(gt_label):
    gt_label = gt_label.cpu()
    ns = gt_label.sum(-1) 
    mask = torch.zeros(gt_label.shape)
    for m, n in enumerate(ns):
        rand_idx = torch.multinomial(torch.tensor([1/n]*int(n)), 1)
        mask_idx = torch.where(gt_label[m] == 1)[0][rand_idx]
        mask[m, mask_idx] = 1
    return mask.float()
