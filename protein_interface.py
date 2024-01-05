import mdtraj as md
import numpy as np
import itertools
from topology_objects import Residue, Atom, store_residue
from utils import resid_from_aidx, ang, index_pairs, ion_index_pairs, aromatic_cation_index_pairs
from hbonds import find_hbond_triplets


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

        # atom indices of all receptor and ligand atoms
        self.idx_receptor = t.top.select(sel_receptor)
        self.idx_ligand = t.top.select(sel_ligand)

        #residue indices of all receptor and ligand residues
        self.resid_receptor = resid_from_aidx(self.t, self.idx_receptor)
        self.resid_ligand = resid_from_aidx(self.t, self.idx_ligand)

        # residue objects for the interface
        interface_resid_receptor, interface_resid_ligand = self.get_interface_residue_idx(method='contacts', cutoff=0.4)
        self.interface_residues_receptor = [store_residue(self.t, x) for x in interface_resid_receptor]
        self.interface_residues_ligand = [store_residue(self.t, x) for x in interface_resid_ligand]

        # atom objects for atoms in interface residues
        interface_atoms_receptor = []
        interface_atoms_ligand = []
        for r in self.interface_residues_receptor:
            for a in r.atoms:
                interface_atoms_receptor.append(a)
        for r in self.interface_residues_ligand:
            for a in r.atoms:
                interface_atoms_ligand.append(a)

        polar_interface_atoms_receptor = [a for a in interface_atoms_receptor if a.is_polar]
        polar_interface_atoms_ligand = [a for a in interface_atoms_ligand if a.is_polar]

        acceptor_interface_atoms_receptor = [a for a in interface_atoms_receptor if a.is_acceptor]
        acceptor_interface_atoms_ligand = [a for a in interface_atoms_ligand if a.is_acceptor]

        donor_interface_atoms_receptor = [a for a in interface_atoms_receptor if a.is_donor]
        donor_interface_atoms_ligand = [a for a in interface_atoms_ligand if a.is_donor]

        cation_interface_atoms_receptor = [a for a in interface_atoms_receptor if a.is_cation]
        cation_interface_atoms_ligand = [a for a in interface_atoms_ligand if a.is_cation]

        anion_interface_atoms_receptor = [a for a in interface_atoms_receptor if a.is_anion]
        anion_interface_atoms_ligand = [a for a in interface_atoms_ligand if a.is_anion]

        # receptor-ligand pairs (and triplets for hbonds)
        self.residue_pairs = index_pairs(self.interface_residues_receptor, self.interface_residues_ligand)
        self.aromatic_residue_pairs = index_pairs(self.interface_residues_receptor, self.interface_residues_ligand, aromatic_residues_only=True)
        self.polar_atom_pairs = index_pairs(polar_interface_atoms_receptor, polar_interface_atoms_ligand)
        self.ion_atom_pairs = ion_index_pairs(cat_rec=cation_interface_atoms_receptor, an_rec=anion_interface_atoms_receptor, cat_lig=cation_interface_atoms_ligand, an_lig=anion_interface_atoms_ligand)
        self.hbond_triplets = find_hbond_triplets(rec_donors=donor_interface_atoms_receptor, rec_acceptors=acceptor_interface_atoms_receptor, lig_donors=donor_interface_atoms_ligand, lig_acceptors=acceptor_interface_atoms_ligand)

        self.aromatic_cation_pairs = aromatic_cation_index_pairs(rec=self.interface_residues_receptor, cat_rec=cation_interface_atoms_receptor, lig=self.interface_residues_ligand, cat_lig=cation_interface_atoms_ligand)

        # contacts
        self.residue_contacts = None
        self.polar_atom_contacts = None
        self.ionic_contacts = None

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

    def get_interface_residue_idx(self, cutoff=0.4, frame=0):
        '''
        Get the residue indices of the interface residues based on contact distance between ligand and receptor residues.

        Parameters
        ----------
        cutoff : float
            Cutoff distance.
        frame : int
            Only one frame of the trajectory is considered. Default 0.
        
        Returns
        -------
        interface_resid_receptor : list of int
            Residue indices of receptor residues in the interface.
        interface_resid_ligand : list of int
            Residue indices of ligand indices in the interface.
        '''
        interface_resid_receptor = []
        interface_resid_ligand = []

        # compute residue contacts of all ligand residue pairs
        residue_pairs = np.array([x for x in itertools.product(self.resid_ligand, self.resid_receptor)])
        contacts, _ = md.compute_contacts(self.t[frame], residue_pairs, scheme='closest-heavy')

        interface_pairs = np.where(contacts[0] < cutoff)[0]

        for i in interface_pairs:
            rp = residue_pairs[i]
            if rp[0] not in interface_resid_ligand:
                interface_resid_ligand.append(rp[0])
            if rp[1] not in interface_resid_receptor:
                interface_resid_receptor.append(rp[1])
        
        return interface_resid_receptor, interface_resid_ligand

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
                angl = np.array([ang(v1, v2) for v1, v2 in zip(o1, o2)])

                dists_pairs[i] = d
                angs_pairs[i] = angl
            
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
