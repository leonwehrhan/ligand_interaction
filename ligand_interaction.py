import mdtraj as md
import numpy as np
import itertools


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

        # store data here
        self.interactions = {}

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
        residue_pairs = np.array([x for x in itertools.product(self.ligand_residues, self.receptor_residues)])

        print(f'Compute residue contacts for {len(residue_pairs)} residue pairs.')

        contacts, _ = md.compute_contacts(self.t, residue_pairs, scheme='closest-heavy')

        self.residue_pairs_idx = residue_pairs
        self.residue_pairs_names = []
        for x in residue_pairs:
            r1 = self.t.top.residue(x[0])
            r2 = self.t.top.residue(x[1])
            s = f'{r1.name}{r1.resSeq}-{r2.name}{r2.resSeq}'
            self.residue_pairs_names.append(s)
        
        self.contact_distances = contacts