from utils import index_pairs
import mdtraj as md
import numpy as np


def get_residue_contacts(t, residue_pairs, cutoff=0.35):
    '''
    Calculate residue contacts, where the closest heavy atoms of residues of ligand and receptor come closer than the cutoff distance.

    Parameters
    ----------
    cutoff : float
        Distance cutoff in nm.
    
    Returns
    -------
    residue_contacts : list of np.ndarray
        List with length n_frames with np.ndarray of shape (n_pairs, 2) for each frame that holds all residue contact pairs of the frame.
    '''
    residue_pairs = index_pairs()

    # initialize list for contacts in each frame
    residue_contacts = []

    # mdtraj residue contacts
    contacts, _ = md.compute_contacts(t, residue_pairs, scheme='closest-heavy')

    for frame in contacts:
        # store residue pairs where contact distance is below cutoff
        pairs = np.array([residue_pairs[i] for i in np.where(frame < cutoff)[0]])
        residue_contacts.append(pairs)

    return residue_contacts


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

