import mdtraj as md
import numpy as np
import itertools
from topology_objects import Residue, Atom
from utils import residue_anions, residue_cations, resid_from_aidx, ang


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
        self.resid_receptor = resid_from_aidx(self.t, self.idx_receptor)
        self.resid_ligand = resid_from_aidx(self.t, self.idx_ligand)

        interface_residues_receptor = None
        interface_residues_ligand = None

        interface_atoms_receptor = None
        interface_atoms_receptor = None

        # interaction data

        # contacts
        self.residue_contacts = None
        self.atom_contacts = None
        self.polar_atom_contacts = None
        self.halogen_contacts = None
        self.ionic_contacts = None

        # contact distances
        self.atom_contact_distances = None

        # h-bonds
        self.hbonds = None

        # aromatic interactions
        self.aromatic_pi_stack = None
        self.aromatic_tshaped = None
        self.aromatic_cation = None

        # dihedrals
        self.dihedrals = None

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')
        print(f'Receptor has {len(self.idx_receptor)} atoms and {len(self.resid_receptor)} residues.')
        print(f'Ligand has {len(self.idx_ligand)} atoms and {len(self.resid_ligand)} residues.')

        # interface residues
        interface_resid_receptor, interface_resid_ligand = self.get_interface(method='contacts', cutoff=0.4)
        self.interface_receptor = [self.store_residue(x) for x in interface_resid_receptor]
        self.interface_ligand = [self.store_residue(x) for x in interface_resid_ligand]
    
    def get_residue_contacts(self, cutoff=0.35, mode='interface'):
        '''
        Calculate residue contacts, where the closest heavy atoms of residues of ligand and receptor come closer than the cutoff distance.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm.
        mode : str
            Only "interface" is implemented.
        
        Returns
        -------
        residue_contacts : list of np.ndarray
            List with length n_frames with np.ndarray of shape (n_pairs, 2) for each frame that holds all residue contact pairs of the frame.
        '''
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
            return residue_contacts

        elif mode == 'all':
            pass
        else:
            pass

    def get_atom_contacts(self, cutoff=0.35, mode='interface', halogen_only=False):
        '''
        Calculate atom contacts, where atoms of ligand and receptor come closer than the cutoff distance.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm.
        mode: str
            Only "interface" is implemented.
        halogen_only : bool
            Only get contacts with ligand halogen atoms. Stored in self.halogen_contacts instead of self.atom_contacts.
        '''
        if mode == 'interface':
            atom_contacts = []
            polar_atom_contacts = []

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
                frame_polar = []
                for i, nl in enumerate(neighbor_lists):
                    ligand_atom_idx = ligand_atoms[i].index
                    nl_frame = nl[i_frame]

                    for receptor_atom_idx in nl_frame:
                        # store atom contact
                        frame.append([ligand_atom_idx, receptor_atom_idx])

                        # store in polar atom contacts if element fits
                        if ligand_atoms[i].element in ['N', 'O', 'S']:
                            
                            # check receptor atom element
                            for a in receptor_atoms:
                                if a.index == receptor_atom_idx:
                                    receptor_atom = a
                            if receptor_atom.element in ['N', 'O', 'S']:
                                # store in polar contacts
                                frame_polar.append([ligand_atom_idx, receptor_atom_idx])

                atom_contacts.append(frame)
                polar_atom_contacts.append(frame_polar)

            
            # store halogen contacts
            if halogen_only:
                self.halogen_contacts = atom_contacts
                return
            
            # store atom contacts in object
            self.atom_contacts = atom_contacts
            self.polar_atom_contacts = polar_atom_contacts

        elif mode == 'all':
            pass
        else:
            pass

    def get_ionic_contacts(self, cutoff=0.37):
        '''
        Calculate contacts between ligand and receptor ions.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm.
        '''
        ionic_contacts = []

        ligand_cations = []
        ligand_anions = []

        for r in self.interface_ligand:
            for a in r.atoms:
                if a.is_cation:
                    ligand_cations.append(a)
                elif a.is_anion:
                    ligand_anions.append(a)
        
        receptor_cations = []
        receptor_anions = []

        for r in self.interface_receptor:
            for a in r.atoms:
                if a.is_cation:
                    receptor_cations.append(a)
                elif a.is_anion:
                    receptor_anions.append(a)
        
        pairs = [x for x in itertools.product([a.index for a in ligand_cations], [a.index for a in receptor_anions])] + [x for x in itertools.product([a.index for a in ligand_anions], [a.index for a in receptor_cations])]
    
        dists = md.compute_distances(self.t, pairs)

        for frame in dists:
            f = []
            for i, x in enumerate(frame):
                if x < cutoff:
                    f.append(pairs[i])
            ionic_contacts.append(f)

        self.ionic_contacts = ionic_contacts
    
    def get_hbonds(self, HA_cutoff=0.25, DHA_cutoff=2.09):
        hbonds = []
        # possible acceptors
        ligand_acceptors = []
        receptor_acceptors = []

        for r in self.interface_ligand:
            for a in r.atoms:
                if a.is_hbond_acceptor:
                    ligand_acceptors.append(a)
        for r in self.interface_receptor:
            for a in r.atoms:
                if a.is_hbond_acceptor:
                    receptor_acceptors.append(a)
            
        # donor, H, acceptor triplets
        triplets = []

        # ligand donors
        for r in self.interface_ligand:
            for a in r.atoms:
                if a.is_hbond_donor:
                    h_atoms = []
                    for b in a.bonds:
                        if b[1] == 'H':
                            h_atoms.append(b[0])
                    for aa in receptor_acceptors:
                        for h in h_atoms:
                            triplets.append([a.index, h, aa.index])

        # receptor donors
        for r in self.interface_receptor:
            for a in r.atoms:
                if a.is_hbond_donor:
                    h_atoms = []
                    for b in a.bonds:
                        if b[1] == 'H':
                            h_atoms.append(b[0])
                    for aa in ligand_acceptors:
                        for h in h_atoms:
                            triplets.append([a.index, h, aa.index])
        
        # HA pairs for distance calculation
        ha_pairs = [x[1:3] for x in triplets]

        # distance and angle calculations
        dists = md.compute_distances(self.t, ha_pairs)
        angs = md.compute_angles(self.t, triplets)

        # boolean masks from distance and angles
        mask_dist = dists < HA_cutoff
        mask_ang = angs < DHA_cutoff  
        mask = mask_dist * mask_ang 

        for frame in mask:
            f = []
            for i, x in enumerate(frame):
                if x:
                    f.append(triplets[i])
            hbonds.append(f)
        
        self.hbonds = hbonds

    def get_aromatic_interactions(self, pi_cutoff=0.40, t_cutoff=0.45, cation_cutoff=0.45, mode='interface'):
        '''
        Calculate aromatic interactions between ligand and receptor. Possible interactions are pi-pi stacking, t-shaped pi interaction or cation-pi interactions.

        Parameters
        ----------
        pi_cutoff : float
            Distance cutoff for pi-pi stacking in nm.
        t_cutoff : float
            Distance cutoff for t-shaped pi interactions in nm.
        cation_cutoff : float
            Distance cutoff for pi-cation interaction in nm.
        mode : str
            Only "interface" is implemented.
        '''
        pi_stacking = []
        t_shaped = []
        pi_cation = []

        # interface residues that have an aromatic system
        ligand_aromatics = [r for r in self.interface_ligand if r.name in ['PHE', 'TYR', 'TRP']]
        receptor_aromatics = [r for r in self.interface_receptor if r.name in ['PHE', 'TYR', 'TRP']]

        # cations in interface residues
        ligand_cations = []
        receptor_cations = []

        for r in self.interface_ligand:
            for a in r.atoms:
                if a.is_cation:
                    ligand_cations.append(a.index)
        
        for r in self.interface_receptor:
            for a in r.atoms:
                if a.is_cation:
                    receptor_cations.append(a.index)

        if mode == 'interface':

            # pairs of ligand-receptor interface residues with aromatics
            pairs = [x for x in itertools.product(range(len(ligand_aromatics)), range(len(receptor_aromatics)))]

            # pairs of aromatics and cations for ligand(Ar)-receptor(CAT) and receptor(Ar)-ligand(CAT)
            pairs_cation_1 = [x for x in itertools.product(range(len(ligand_aromatics)), range(len(receptor_cations)))]
            pairs_cation_2 = [x for x in itertools.product(range(len(receptor_aromatics)), range(len(ligand_cations)))]
        
            # store distances and orth angles for all aromatic pairs
            dists_pairs = np.zeros((len(pairs), self.t.n_frames))
            angs_pairs = np.zeros((len(pairs), self.t.n_frames))

            for i, pair in enumerate(pairs):
                r1 = ligand_aromatics[pair[0]]
                r2 = receptor_aromatics[pair[1]]

                c1, o1 = self.aromatic_centroid_orth(r1)
                c2, o2 = self.aromatic_centroid_orth(r2)

                d = np.linalg.norm(c1 - c2, axis=1)
                ang = np.array([self._ang(v1, v2) for v1, v2 in zip(o1, o2)])

                dists_pairs[i] = d
                angs_pairs[i] = ang
            
            for i_frame in range(self.t.n_frames):
                pi_frame = []
                t_frame = []
                for i_pair in range(len(pairs)):
                    if dists_pairs[i_pair][i_frame] < pi_cutoff:
                        if angs_pairs[i_pair][i_frame] < 0.17 or angs_pairs[i_pair][i_frame] > 2.97:
                            pi_frame.append(pairs[i_pair])
                    
                    if dists_pairs[i_pair][i_frame] < t_cutoff:
                        if angs_pairs[i_pair][i_frame] < 1.75 and angs_pairs[i_pair][i_frame] > 1.40:
                            t_frame.append(pairs[i_pair])
                
                pi_stacking.append(pi_frame)
                t_shaped.append(t_frame)
            
            # pi cation contacts
            dist_cation_pairs_1 = np.zeros((len(pairs_cation_1), self.t.n_frames))
            dist_cation_pairs_2 = np.zeros((len(pairs_cation_2), self.t.n_frames))

            for i, pair in enumerate(pairs_cation_1):
                r_ar = ligand_aromatics[pair[0]]
                a_cat = receptor_cations[pair[1]]

                c_ar, o_ar = self.aromatic_centroid_orth(r_ar)

                d = np.linalg.norm(c_ar - self.t.xyz[:, a_cat], axis=1)
                dist_cation_pairs_1[i] = d

            for i, pair in enumerate(pairs_cation_2):
                r_ar = receptor_aromatics[pair[0]]
                a_cat = ligand_cations[pair[1]]

                c_ar, o_ar = self.aromatic_centroid_orth(r_ar)

                d = np.linalg.norm(c_ar - self.t.xyz[:, a_cat], axis=1)
                dist_cation_pairs_2[i] = d
            
            for i_frame in range(self.t.n_frames):
                cat_frame = []

                for i_pair in range(len(pairs_cation_1)):
                    if dist_cation_pairs_1 < cation_cutoff:
                        cat_frame.append(pairs_cation_1[i_pair])
                    
                for i_pair in range(len(pairs_cation_2)):
                    if dist_cation_pairs_2 < cation_cutoff:
                        cat_frame.append(pairs_cation_2[i_pair])
                
                pi_cation.append(cat_frame)

        self.aromatic_pi_stack = pi_stacking
        self.aromatic_tshaped = t_shaped
        self.aromatic_cation = pi_cation



    def get_atom_contact_distances(self):
        '''
        Calculate distances for all atom contact pairs. Atom contacts have to be calculated before calling this method.
        '''
        atom_contact_distances = {}
        unique_atom_contacts = []
        for frame in self.atom_contacts:
            for c in frame:
                if c not in unique_atom_contacts:
                    unique_atom_contacts.append(c)
        
        dists = md.compute_distances(self.t, unique_atom_contacts)

        for i, c in enumerate(unique_atom_contacts):
            s = f'{self.t.top.atom(c[0])}--{self.t.top.atom(c[1])}'
            atom_contact_distances[s] = dists[:, i]
        
        self.atom_contact_distances = atom_contact_distances

    def get_dihedrals(self, mode='interface'):
        '''
        Calculate all backbone and sidechain dihedral angles in the trajectory.
        '''
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
        
        # interface residues only
        if mode == 'interface':
            dihedrals_interface = {}
            resid_interface = []

            for r in self.interface_receptor:
                resid_interface.append(r.index)
            for r in self.interface_ligand:
                resid_interface.append(r.index)

            for dihed in dihedrals:
                dihedrals_interface[dihed] = {}
                for resid in dihedrals[dihed]:
                    if resid in resid_interface:
                        dihedrals_interface[dihed][resid] = dihedrals[dihed][resid]
            
            self.dihedrals = dihedrals_interface
            return

        
        # store dihedrals
        self.dihedrals = dihedrals
    
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

    def aromatic_centroid_orth(self, r):
        '''
        Get centroid and orthogonal vector of aromatic system in residue.

        Parameters
        ----------
        r : Residue
            Residue object with aromatic system. Has to be PHE, TYR or TRP.
        
        Returns
        -------
        centroid_coordinates : np.ndarray
            Array of shape (n_frames, 3) that holds coordinates of aromatic centroid throughout the simulation.
        orthogonal_vectors : np.ndarray
            Array of shape (n_frames, 3) that holds the orthogonal vectors of the aromatic system throughout the simulation.
        '''
        aromatic_atom_idx = []
        plane_idx = []

        centroid_coordinates = np.zeros((self.t.n_frames, 3))
        orthogonal_vectors = [(self.t.n_frames, 3)]

        # get aromatic atom indices based on atom name (in amber14sb)
        if r.name in ['PHE', 'TYR']:
            for a in r.atoms:
                if a.name in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']:
                    aromatic_atom_idx.append(a.index)
        elif r.name == 'TRP':
            for a in r.atoms:
                if a.name in ['CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2']:
                    aromatic_atom_idx.append(a.index)
        
        # get three atoms in aromatic system to define plane for orthogonal
        if r.name in ['PHE', 'TYR']:
            for a in r.atoms:
                if a.name in ['CG', 'CE1', 'CE2']:
                    plane_idx.append(a.index)
        elif r.name == 'TRP':
            for a in r.atoms:
                if a.name in ['CG', 'CZ2', 'CZ3']:
                    plane_idx.append(a.index)
        
        # coordinates of armotaic atoms
        xyz_ar = self.t.xyz[:, aromatic_atom_idx]
        
        # calculate centroid coordinates
        for i, x in enumerate(xyz_ar):
            centroid_coordinates[i, :] = np.mean(x, axis=0)
        
        # calculate orthogonal vectors
        vec1 = self.t.xyz[:, plane_idx[1]] - self.t.xyz[:, plane_idx[0]]
        vec2 = self.t.xyz[:, plane_idx[2]] - self.t.xyz[:, plane_idx[0]]

        for i in range(self.t.n_frames):
            dot = np.dot(vec1[i], vec2[i])
            orthogonal_vectors[i, :] = dot / np.linalg.norm(dot)
        
        return centroid_coordinates, orthogonal_vectors


if __name__ == 'main':
    pass
