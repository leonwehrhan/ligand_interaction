import numpy as np


# cations and anions in aminoacids in AMBER14SB
RESIDUE_CATIONS = {'ARG':['NH1', 'NH2'], 'LYS':['NZ'], 'HIP':['ND1', 'NE2'], 'HIE':['NE2'], 'HID':['ND1']}
RESIDUE_ANIONS = {'ASP':['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']}

# aromatic ring atoms in amino acids in AMBER14SB
AROMATIC_RING_ATOMS = {'PHE':['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'TYR':['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'TRP':['CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2']}

HALOGEN_ELEMENTS = ['F', 'Cl', 'Br', 'I']
POLAR_ELEMENTS = ['N', 'O', 'S']


def ang(v1, v2):
    '''
    Calculate angle between two vectors.
    '''
    u_v1 = v1 / np.linalg.norm(v1)
    u_v2 = v2 / np.linalg.norm(v2)

    ang = np.arccos(np.clip(np.dot(u_v1, u_v2), -1.0, 1.0))
    return ang


def resid_from_aidx(t, atom_idx):
    resid = []
    for i in atom_idx:
        a = t.top.atom(i)
        r_i = a.residue.index
        if r_i not in resid:
            resid.append(r_i)
    return np.array(resid)