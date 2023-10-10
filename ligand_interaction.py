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
        print(f'Ligand has {len(self.idx_ligand)} atoms and {len(self.resid_ligand)} residues.')

        # interface residues
        self.interface_receptor, self.interface_ligand = self.get_interface(method='contacts', cutoff=0.35)

    
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
    
    def store_residue(self, resid):
        r = self.t.top.residue(resid)
        R = Residue()
        R.index = resid
        R.name = r.name
        R.resSeq = r.resSeq

        bonds = []
        for b in self.t.top.bonds:
            if b[0].residue == r or b[1].residue == r:
                bonds.append([b[0].index, b[1].index])
        R.bonds = bonds

        atoms = []
        for a in r.atoms:
            A = Atom()
            A.index = a.index
            A.name = a.name
            A.element = a.element
            A.residue = R

            a_bonds = []
            for b in self.t.top.bonds:
                if b[0].index == a.index or b[1].index == a.index:
                    for x in b:
                        if x.index != a.index:
                            a_bonds.append([x.index, x.element])
            A.bonds = a_bonds

            A.is_sidechain = a.is_sidechain

            atoms.append(A)


class Residue:
    def __init__(self):
        self.index = None
        self.name = None
        self.resSeq = None

        self.atoms = []
        self.bonds = []


class Atom:
    def __init__(self):
        self.index = None
        self.name = None
        self.element = None
        self.residue = None

        self.bonds = []

        self.is_sidechain = None
        self.is_hbond_donor = None
        self.is_hbond_acceptor = None
        self.is_cation = None
        self.is_anion = None
        self.is_halogen = None
        self.is_hydrophobic = None


if __name__ == 'main':
    pass
