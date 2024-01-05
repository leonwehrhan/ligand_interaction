from context import topology_objects
import mdtraj as md


def test_store_residue():
    t = md.load('data/test.xtc', top='data/test.pdb')
    R = topology_objects.store_residue(t, 237)

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
        
        if a.name == 'N':
            assert a.is_hbond_donor == True
            assert a.is_hbond_acceptor == True
        
        if a.name == 'O':
            assert a.is_hbond_acceptor == True
            assert a.is_hbond_donor == False
        
        if a.name == 'CG':
            assert a.is_hbond_acceptor == False
            assert a.is_hbond_donor == False