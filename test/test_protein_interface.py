from context import protein_interface
import mdtraj as md
import numpy as np
import pytest


@pytest.fixture
def interface_object():
    t = md.load('data/test.xtc', top='data/test.pdb')

    sel_receptor = 'resid 0 to 222'
    sel_ligand = 'resid 223 to 280'

    I = protein_interface.Interface(t, sel_receptor=sel_receptor, sel_ligand=sel_ligand)

    return I

def test_Interface_init(interface_object):
    I = interface_object

def test_resids(interface_object):
    I = interface_object
    t = md.load('data/test.xtc', top='data/test.pdb')

    idx_rec = I.idx_receptor
    idx_lig = I.idx_ligand

    np.testing.assert_array_equal(idx_rec, t.top.select('resid 0 to 222'))
    np.testing.assert_array_equal(idx_lig, t.top.select('resid 223 to 280'))

    np.testing.assert_array_equal(I.resid_receptor, np.unique(I.resid_receptor))
    np.testing.assert_array_equal(I.resid_ligand, np.unique(I.resid_ligand))

def test_interface_residues(interface_object):
    I = interface_object

    for r in I.interface_residues_receptor:
        assert r.index in I.resid_receptor

    for r in I.interface_residues_ligand:
        assert r.index in I.resid_ligand
    
    # for cutoff=0.5
    assert [x.index for x in I.interface_residues_receptor].sort() == [173, 80, 191, 192, 193, 39, 83, 171, 172, 174, 175, 176, 190, 196, 23, 24, 40, 21, 22, 130, 77, 78, 154].sort()
    assert [x.index for x in I.interface_residues_ligand].sort() == [233, 234, 235, 236, 237, 238, 239, 240, 241, 256, 258, 259, 260, 261].sort()