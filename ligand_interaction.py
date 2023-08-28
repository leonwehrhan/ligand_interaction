import mdtraj as md
import numpy as np


class Interaction:
    '''
    Class that holds interaction calculations.

    Attributes
    ----------
    t : md.Trajectory
        Input trajectory.
    sel_receptor : str
        Selection string for receptor.
    sel_ligand : str
        Selection string for ligand.
    '''
    def __init__(self, t, sel_receptor, sel_ligand):
        self.t = t
        self.idx_receptor = t.top.select(sel_receptor)
        self.idx_ligand = t.top.select(sel_ligand)

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')
        print(f'Receptor has {len(self.idx_receptor)} atoms.')
        print(f'Ligand has {len(self.idx_ligands)} atoms.')

        # residue indices of ligand residues
        ligand_residues = []
        for i in self.idx_ligand:
            a = t.top.atom(i)
            r_i = a.residue.index
            if r_i not in ligand_residues:
                ligand_residues.append(r_i)
        self.ligand_residues = ligand_residues

        # residue indices of receptor residues
        receptor_residues = []
        for i in self.idx_receptor:
            a = t.top.atom(i)
            r_i = a.residue.index
            if r_i not in receptor_residues:
                receptor_residues.append(r_i)
        self.receptor_residues = receptor_residues

        print(f'Receptor has {len(receptor_residues)} residues.')
        print(f'Ligand has {len(ligand_residues)} residues.')

    def residue_contacts(self):
        pass