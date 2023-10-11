from context import ligand_interaction
import mdtraj as md
import numpy as np
import pytest


@pytest.fixture
def interface_object():
    t = md.load('test.xtc', top='test.pdb')

    sel_receptor = 'resid 0 to 222'
    sel_ligand = 'resid 223 to 280'

    I = ligand_interaction.Interface(t, sel_receptor=sel_receptor, sel_ligand=sel_ligand)

    return I

def test_Interface_init(interface_object):
    I = interface_object

def test_resids(interface_object):
    I = interface_object
    t = md.load('test.xtc', top='test.pdb')

    idx_rec = I.idx_receptor
    idx_lig = I.idx_ligand

    np.testing.assert_array_equal(idx_rec, t.top.select('resid 0 to 222'))
    np.testing.assert_array_equal(idx_lig, t.top.select('resid 223 to 280'))

    np.testing.assert_array_equal(I.resid_receptor, np.unique(I.resid_receptor))
    np.testing.assert_array_equal(I.resid_ligand, np.unique(I.resid_ligand))

