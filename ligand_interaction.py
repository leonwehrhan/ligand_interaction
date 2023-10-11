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
        interface_resid_receptor, interface_resid_ligand = self.get_interface(method='contacts', cutoff=0.35)
        self.interface_receptor = [self.store_residue(x) for x in interface_resid_receptor]
        self.interface_ligand = [self.store_residue(x) for x in interface_resid_ligand]
    
    def resid_from_aidx(self, atom_idx):
        resid = []
        for i in atom_idx:
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
            A.element = a.element.symbol
            A.residue = R

            a_bonds = []
            for b in self.t.top.bonds:
                if b[0].index == a.index or b[1].index == a.index:
                    for x in b:
                        if x.index != a.index:
                            a_bonds.append([x.index, x.element.symbol])
            A.bonds = a_bonds

            A.is_sidechain = a.is_sidechain

            if A.element == 'N' or A.element == 'O':
                A.is_hbond_acceptor = True
            else:
                A.is_hbond_acceptor = False
            
            if A.element == 'N' or A.element == 'O':
                if any([x[1] == 'H' for x in A.bonds]):
                    A.is_hbond_donor = True
                else:
                    A.is_hbond_donor = False
            else:
                A.is_hbond_donor = False
            
            if A.element in ['F', 'Cl', 'Br', 'I']:
                A.is_halogen = True
            else:
                A.is_halogen = False
            
            if R.name in residue_anions:
                if a.name in residue_anions[R.name]:
                    A.is_anion = True
                else:
                    A.is_anion = False
            else:
                A.is_anion = False

            if R.name in residue_cations:
                if a.name in residue_cations[R.name]:
                    A.is_cation = True
                else:
                    A.is_cation = False
            else:
                A.is_cation = False

            atoms.append(A)

        R.atoms = atoms

        return R


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


residue_cations = {'ARG':['NH1', 'NH2'], 'LYS':['NZ'], 'HIP':['ND1', 'NE2'], 'HIE':['NE2'], 'HID':['ND1']}
residue_anions = {'ASP':['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']}


if __name__ == 'main':
    pass
