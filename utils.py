import numpy as np


residue_cations = {'ARG':['NH1', 'NH2'], 'LYS':['NZ'], 'HIP':['ND1', 'NE2'], 'HIE':['NE2'], 'HID':['ND1']}
residue_anions = {'ASP':['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']}


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