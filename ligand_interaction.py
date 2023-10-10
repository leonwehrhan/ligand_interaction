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
    '''
    def __init__(self, t, sel_receptor, sel_ligand):
        self.t = t

        # atom indices of receptor and ligand
        self.idx_receptor = t.top.select(sel_receptor)
        self.idx_ligand = t.top.select(sel_ligand)

        #residue indices of receptor and ligand
        self.resid_receptor = self.resid_from_aidx(self.idx_receptor)
        self.resid_ligand = self.resid_from_aidx(self.idx_ligand)

        # interaction data
        self.residue_contacts = None

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')
        print(f'Receptor has {len(self.idx_receptor)} atoms and {len(self.resid_receptor)} residues.')
        print(f'Ligand has {len(self.idx_ligands)} atoms and {len(self.resid_ligand)} residues.')

        # interface residues
        self.interface_receptor, self.interface_ligand = self.get_interface(self, method='contacts', cutoff=0.35)

    
    def resid_from_aidx(self, atom_idx):
        resid = []
        for i in self.idx_ligand:
            a = self.t.top.atom(i)
            r_i = a.residue.index
            if r_i not in resid:
                resid.append(r_i)
        return resid
    
    def get_interface(self, method='contacts', cutoff=0.35):
        interface_resid_receptor = []
        interface_resid_ligand = []
        if method == 'contacts':
            # compute residue contacts of all ligand residue pairs
            # only apply first frame of trajectory
            residue_pairs = np.array([x for x in itertools.product(self.resid_ligand, self.resid_receptor)])
            contacts, _ = md.compute_contacts(self.t[0], residue_pairs, scheme='closest-heavy')

            interface_pairs = np.where(contacts[0] < cutoff)[0]

            for i in interface_pairs:
                rp = residue_pairs[i]
                if rp[0] not in interface_resid_ligand:
                    interface_resid_ligand.append(rp[0])
                if rp[1] not in interface_resid_receptor:
                    interface_resid_receptor.append(rp[1])
            
            return interface_resid_receptor, interface_resid_ligand
        else:
            pass


if __name__ == 'main':
    pass
