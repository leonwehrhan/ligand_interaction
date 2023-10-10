import mdtraj as md
import numpy as np
import itertools


class Interface:
    '''
    Holds information about the protein-protein (or protein-ligand) interface.

    Attributes
    ----------
    t : md.Trajectory
        Input trajectory.
    sel_receptor : str
        Selection string for receptor.
    sel_ligand : str
        Selection string for ligand.
    method : str
        Method for determining interface.
    '''
    def __init__(self, t, sel_receptor, sel_ligand, method='contacts'):
        self.t = t

        # atom indices of receptor and ligand
        self.idx_receptor = t.top.select(sel_receptor)
        self.idx_ligand = t.top.select(sel_ligand)

        #residue indices of receptor and ligand
        self.resid_receptor = self.resid_from_aidx(self.idx_receptor)
        self.resid_ligand = self.resid_from_aidx(self.idx_ligand)

        self.interface_residues = []

        # store interaction data here
        self.interactions = {}

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')
        print(f'Receptor has {len(self.idx_receptor)} atoms and {len(self.resid_receptor)} residues.')
        print(f'Ligand has {len(self.idx_ligands)} atoms and {len(self.resid_ligand)} residues.')

    
    def resid_from_aidx(self, atom_idx):
        resid = []
        for i in self.idx_ligand:
            a = self.t.top.atom(i)
            r_i = a.residue.index
            if r_i not in resid:
                resid.append(r_i)
        return resid
    
    def get_interface(self, method='contacts'):
        pass

    def residue_contacts(self, cutoff=0.35):
        '''
        Compute residue contact distances of all ligand-receptor residue pairs. Has to be run to determine interface residues.
        '''
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
        
        # store contact distances
        self.residue_contact_distances = contacts
