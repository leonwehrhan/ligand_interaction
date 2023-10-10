from context import ligand_interaction
import mdtraj as md


def test_Interface_init():
    t = md.load('test.xtc', top='test.pdb')

    sel_receptor = 'resid 0 to 222'
    sel_ligand = 'resid 223 to 280'

    I = ligand_interaction.Interface(t, sel_receptor=sel_receptor, sel_ligand=sel_ligand)