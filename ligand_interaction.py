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
        self.atom_contacts = None
        self.halogen_contacts = None
        self.dihedrals = None

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')
        print(f'Receptor has {len(self.idx_receptor)} atoms and {len(self.resid_receptor)} residues.')
        print(f'Ligand has {len(self.idx_ligand)} atoms and {len(self.resid_ligand)} residues.')

        # interface residues
        interface_resid_receptor, interface_resid_ligand = self.get_interface(method='contacts', cutoff=0.4)
        self.interface_receptor = [self.store_residue(x) for x in interface_resid_receptor]
        self.interface_ligand = [self.store_residue(x) for x in interface_resid_ligand]
    
    def get_residue_contacts(self, cutoff=0.35, mode='interface'):
        if mode == 'interface':
            # residue indices for interface residues
            interface_resid_receptor = [x.index for x in self.interface_receptor]
            interface_resid_ligand = [x.index for x in self.interface_ligand]

            # pairs of all interface receptor and ligand residues
            residue_pairs = np.array([x for x in itertools.product(interface_resid_receptor, interface_resid_ligand)])

            # initialize list for contacts in each frame
            residue_contacts = []

            # mdtraj residue contacts
            contacts, _ = md.compute_contacts(self.t, residue_pairs, scheme='closest-heavy')

            for frame in contacts:
                # store residue pairs where contact distance is below cutoff
                pairs = np.array([residue_pairs[i] for i in np.where(frame < cutoff)[0]])
                residue_contacts.append(pairs)

            # store contacts in object
            self.residue_contacts = residue_contacts

        elif mode == 'all':
            pass
        else:
            pass

    def get_atom_contacts(self, cutoff=0.35, mode='interface', halogen_only=False):
        if mode == 'interface':
            atom_contacts = []

            # atom objects for ligand and receptor heavy atoms
            ligand_atoms = []
            receptor_atoms = []

            # all heavy atoms of ligand interface
            for r in self.interface_ligand:
                for a in r.atoms:
                    if a.element != 'H':
                        ligand_atoms.append(a)
            
            # all heavy atoms of receptor interface
            for r in self.interface_receptor:
                for a in r.atoms:
                    if a.element != 'H':
                        receptor_atoms.append(a)
            
            # for halogen contacts remove all atoms but halogens from ligand atoms
            if halogen_only:
                ligand_atoms = [a for a in ligand_atoms if a.is_halogen]

            # store neighbor list for each ligand atom
            neighbor_lists = []
            
            # receptor heavy atom indices as haystack_indices
            idx_receptor = [a.index for a in receptor_atoms]

            for a in ligand_atoms:
                # compute_neighbors for ligand atom contacts
                neigh = md.compute_neighbors(self.t, cutoff=cutoff, query_indices=[a.index], haystack_indices=idx_receptor)
                neighbor_lists.append(neigh)
            
            for i_frame in range(len(neighbor_lists[0])):
                frame = []
                for i, nl in enumerate(neighbor_lists):
                    ligand_atom_idx = ligand_atoms[i].index
                    nl_frame = nl[i_frame]

                    for residue_atom_idx in nl_frame:
                        frame.append([ligand_atom_idx, residue_atom_idx])
                atom_contacts.append(frame)
            
            # store halogen contacts
            if halogen_only:
                self.halogen_contacts = atom_contacts
                return
            # store atom contacts in object
            self.atom_contacts = atom_contacts

        elif mode == 'all':
            pass
        else:
            pass

    def get_dihedrals(self, mode='interface'):
        dihedrals = {}

        # phi backbone dihedrals
        dihedrals['phi'] = {}
        idx, phi = md.compute_phi(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[2]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['phi'][resid] = phi[:, i]
        
        # psi backbone dihedrals
        dihedrals['psi'] = {}
        idx, psi = md.compute_psi(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['psi'][resid] = psi[:, i]

        # chi1 sidechain dihedrals
        dihedrals['chi1'] = {}
        idx, chi1 = md.compute_chi1(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['chi1'][resid] = chi1[:, i]

        # chi2 sidechain dihedrals
        dihedrals['chi2'] = {}
        idx, chi2 = md.compute_chi2(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['chi2'][resid] = chi2[:, i]
    
        # chi3 sidechain dihedrals
        dihedrals['chi3'] = {}
        idx, chi3 = md.compute_chi3(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['chi3'][resid] = chi3[:, i]

        # chi4 sidechain dihedrals
        dihedrals['chi4'] = {}
        idx, chi4 = md.compute_chi4(self.t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = self.t.top.atom(idx_CA).residue.index
            dihedrals['chi4'][resid] = chi4[:, i]
        
        # store dihedrals
        self.dihedrals = dihedrals

    def resid_from_aidx(self, atom_idx):
        resid = []
        for i in atom_idx:
            a = self.t.top.atom(i)
            r_i = a.residue.index
            if r_i not in resid:
                resid.append(r_i)
        return np.array(resid)
    
    def get_interface(self, method='contacts', cutoff=0.4):
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
        # basic residue information
        r = self.t.top.residue(resid)
        R = Residue()
        R.index = resid
        R.name = r.name
        R.resSeq = r.resSeq

        # all bond that involve residue atoms
        bonds = []
        for b in self.t.top.bonds:
            if b[0].residue == r or b[1].residue == r:
                bonds.append([b[0].index, b[1].index])
        R.bonds = bonds

        # list of atoms
        atoms = []
        for a in r.atoms:
            # basic atom information
            A = Atom()
            A.index = a.index
            A.name = a.name
            A.element = a.element.symbol
            A.residue = R

            # list of tuple (index, element) of atoms the atom is bonded to
            a_bonds = []
            for b in self.t.top.bonds:
                if b[0].index == a.index or b[1].index == a.index:
                    for x in b:
                        if x.index != a.index:
                            a_bonds.append([x.index, x.element.symbol])
            A.bonds = a_bonds

            # is_sidechain from mdtraj
            A.is_sidechain = a.is_sidechain

            # elements O and N are possible H-bond acceptors
            if A.element == 'N' or A.element == 'O':
                A.is_hbond_acceptor = True
            else:
                A.is_hbond_acceptor = False
            
            # if element H is bonded to N or O means possible donor
            if A.element == 'N' or A.element == 'O':
                if any([x[1] == 'H' for x in A.bonds]):
                    A.is_hbond_donor = True
                else:
                    A.is_hbond_donor = False
            else:
                A.is_hbond_donor = False
            
            # halogen element symbols
            if A.element in ['F', 'Cl', 'Br', 'I']:
                A.is_halogen = True
            else:
                A.is_halogen = False
            
            # check list of amino acid anion atoms (based on residue name)
            if R.name in residue_anions:
                if a.name in residue_anions[R.name]:
                    A.is_anion = True
                else:
                    A.is_anion = False
            else:
                A.is_anion = False

            # check list of amino acid cation atoms (based on residue name)
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
