import mdtraj as md
import numpy as np
import itertools
from topology_objects import Residue, Atom, store_residue
from utils import resid_from_aidx, ang, index_pairs, ion_index_pairs, aromatic_cation_index_pairs
from hbonds import find_hbond_triplets
from aromatics import aromatic_centroid_orth


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

        print(f'Analyzing interactions in trajectory with {t.n_frames} frames and {t.n_atoms} atoms.')

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
        self.residue_contacts = np.zeros(len(self.t), len(self.residue_pairs), dtype=bool)
        self.polar_atom_contacts = np.zeros(len(self.t), len(self.polar_atom_pairs), dtype=bool)
        self.ionic_contacts = np.zeros(len(self.t), len(self.ion_atom_pairs), dtype=bool)

        # h-bonds
        self.hbonds = None

        # aromatic interactions
        self.aromatic_pi_stack = None
        self.aromatic_tshaped = None
        self.aromatic_cation = None

        # dihedrals
        self.dihedrals = None

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
    
    def get_residue_contacts(self, cutoff=0.35):
        '''
        Calculate residue contacts, where the closest heavy atoms of residues of ligand and receptor come closer than the cutoff distance.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm.
        '''
        residue_pairs = self.residue_pairs
        contacts, _ = md.compute_contacts(self.t, residue_pairs, scheme='closest-heavy')
        self.residue_contacts = contacts < cutoff

    def get_atom_contacts(self, pairs, cutoff=0.37):
        '''
        Calculate contacts between atoms.

        Parameters
        ----------
        pairs : list of tuple of int
            Atom pairs.
        cutoff : float
            Distance cutoff in nm.
        '''
        dists = md.compute_distances(self.t, pairs)
        return dists < cutoff
    
    def get_hbonds(self, triplets):
        '''
        Calculate H bonds based on D-H-A triplets and the Baker Hubbard criterion.
        '''
        HA_cutoff = 0.25
        DHA_cutoff = 2.09
        
        # HA pairs for distance calculation
        ha_pairs = [x[1:3] for x in triplets]

        # distance and angle calculations
        dists = md.compute_distances(self.t, ha_pairs)
        angs = md.compute_angles(self.t, triplets)

        # boolean masks from distance and angles
        mask_dist = dists < HA_cutoff
        mask_ang = angs < DHA_cutoff  
        mask = mask_dist * mask_ang 
        return mask
    
    def get_cation_pi(self, pairs, cutoff=0.45):
        '''
        Calculate cation-pi interactions.
        '''
        contacts = np.zeros((len(self.t), len(pairs)), dtype=bool)
        for i, p in enumerate(pairs):
            c_ar, o_ar = self.aromatic_centroid_orth(p[0])
            c_cat = self.t.xyz[:, p[1]]
            d = np.linalg.norm(c_ar - c_cat, axis=1)
            contacts[:, i] = d < cutoff
        return contacts

        

if __name__ == 'main':
    pass
