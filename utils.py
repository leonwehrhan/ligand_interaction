import numpy as np
import itertools


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


def index_pairs(x_rec, x_lig, aromatic_residues_only=False):
    '''
    Make pairs of indices from two lists of Atom or Residue objects, where each object of one list is paired with all of the other.

    Parameters
    ----------
    x_rec : list of Atom or list of Residue
        Atom/Residue objects of receptor.
    x_lig : list of Atom or list of Residue
        Atom/Residue objects of ligand.
    aromatic_residues_only : bool
        Only return pairs of aromatic residues.
    
    Returns
    -------
    pairs : list of tuple of int
        Paired indices.
    '''
    if not aromatic_residues_only:
        idx_rec = [x.index for x in x_rec]
        idx_lig = [x.index for x in x_lig]
    else:
        idx_rec = [x.index for x in x_rec if x.name in AROMATIC_RING_ATOMS]
        idx_lig = [x.index for x in x_lig if x.name in AROMATIC_RING_ATOMS]

    pairs = [x for x in itertools.product(idx_rec, idx_lig)]
    return pairs


def ion_index_pairs(cat_rec, an_rec, cat_lig, an_lig):
    '''
    Make pairs of ionic atom indices. The receptor cations are paired with ligand anions and vice versa.

    Parameters
    ----------
    cat_rec : list of Atom
        Cations of receptor.
    an_rec : list of Atom
        Anions of receptor.
    cat_lig : list of Atom
        Cations of ligand.
    an_lig : list of Atom
        Anions of ligand.

    Returns
    -------
    pairs : list of tuple of int
    '''
    idx_cat_rec = [x.index for x in cat_rec]
    idx_an_rec = [x.index for x in an_rec]
    idx_cat_lig = [x.index for x in cat_lig]
    idx_an_lig = [x.index for x in an_lig]

    pairs = [x for x in itertools.product(idx_cat_rec, idx_an_lig)] + [x for x in itertools.product(idx_an_rec, idx_cat_lig)]
    return pairs


def aromatic_cation_index_pairs(rec, cat_rec, lig, cat_lig):
    '''
    Make pairs of aromatic residues and cation atom indices. Here, the residue objects are paired with the cation atom indices.

    Parameters
    ----------
    rec : list of Residue
        Residues of receptor interface.
    cat_rec : list of Atom
        Cationic atoms of receptor. 
    lig : list of Residue
        Residues of ligand interface.
    cat_lig : list of Atom
        Cationic atoms of ligand. 
    
    Returns
    -------
    pairs : list of tuple (Residue, int)
    '''
    pairs = [x for x in itertools.product(rec, cat_lig)] + [x for x in itertools.product(lig, cat_rec)]
    return pairs