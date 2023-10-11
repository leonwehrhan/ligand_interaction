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

def test_interface_residues(interface_object):
    I = interface_object

    for r in I.interface_receptor:
        assert r.index in I.resid_receptor

    for r in I.interface_ligand:
        assert r.index in I.resid_ligand
    
    # for cutoff=0.4
    assert [x.index for x in I.interface_receptor].sort() == [173, 192, 193, 39, 80, 191, 171, 172, 174, 175, 176, 23, 24, 21, 22, 130, 77, 78].sort()
    assert [x.index for x in I.interface_ligand].sort() == [233, 235, 236, 237, 238, 239, 240, 241, 256, 258, 259, 260, 261].sort()

def test_store_residue(interface_object):
    I = interface_object
    R = I.store_residue(237)

    assert R.index == 237
    assert R.name == 'E3G'
    assert R.resSeq == 15

    for a in R.atoms:
        assert a.index in range(3433, 3446)
    
    for a in R.atoms:
        assert a.name in ['N', 'H', 'CA', 'HA', 'CB', 'HB1', 'HB2', 'CG', 'F', 'F1', 'F2', 'C', 'O']
        assert a.element in ['N', 'H', 'C', 'F', 'O']

        if a.element == 'F':
            assert a.is_halogen == True
        
        if a.name == 'N':
            assert [3434, 'H'] in a.bonds
            assert [3435, 'C'] in a.bonds
        
        if a.name == 'CG':
            assert [3441, 'F'] in a.bonds
            assert [3442, 'F'] in a.bonds
            assert [3443, 'F'] in a.bonds