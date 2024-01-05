import numpy as np
from utils import AROMATIC_RING_ATOMS, AROMATIC_PLANE


def aromatic_centroid_orth(t, r):
    '''
    Get centroid and orthogonal vector of aromatic system in residue.

    Parameters
    ----------
    t : md.Trajectory
        Trajectory object.
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

    centroid_coordinates = np.zeros((t.n_frames, 3))
    orthogonal_vectors = [(t.n_frames, 3)]

    # get aromatic atom indices based on atom name (in amber14sb)
    for a in r.atoms:
        if a.name in AROMATIC_RING_ATOMS[r.name]:
            aromatic_atom_idx.append(a.index)
    
    # get three atoms in aromatic system to define plane for orthogonal
    for a in r.atoms:
        if a.name in AROMATIC_PLANE[r.name]:
            plane_idx.append(a.index)
    
    # coordinates of armotaic atoms
    xyz_ar = t.xyz[:, aromatic_atom_idx]
    
    # calculate centroid coordinates
    for i, x in enumerate(xyz_ar):
        centroid_coordinates[i, :] = np.mean(x, axis=0)
    
    # calculate orthogonal vectors
    vec1 = t.xyz[:, plane_idx[1]] - t.xyz[:, plane_idx[0]]
    vec2 = t.xyz[:, plane_idx[2]] - t.xyz[:, plane_idx[0]]

    for i in range(t.n_frames):
        dot = np.dot(vec1[i], vec2[i])
        orthogonal_vectors[i, :] = dot / np.linalg.norm(dot)
    
    return centroid_coordinates, orthogonal_vectors


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
