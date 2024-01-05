import mdtraj as md


def trj_dihedrals(t, mode='all', interface_res_rec=None, interface_res_lig=None):
    '''
    Calculate all backbone and sidechain dihedral angles in the trajectory.
    '''
    if mode == 'all':
        dihedrals = {}
        
        # phi backbone dihedrals
        dihedrals['phi'] = {}
        idx, phi = md.compute_phi(t)
        for i, d in enumerate(idx):
            idx_CA = d[2]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['phi'][resid] = phi[:, i]
        
        # psi backbone dihedrals
        dihedrals['psi'] = {}
        idx, psi = md.compute_psi(t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['psi'][resid] = psi[:, i]

        # chi1 sidechain dihedrals
        dihedrals['chi1'] = {}
        idx, chi1 = md.compute_chi1(t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['chi1'][resid] = chi1[:, i]

        # chi2 sidechain dihedrals
        dihedrals['chi2'] = {}
        idx, chi2 = md.compute_chi2(t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['chi2'][resid] = chi2[:, i]

        # chi3 sidechain dihedrals
        dihedrals['chi3'] = {}
        idx, chi3 = md.compute_chi3(t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['chi3'][resid] = chi3[:, i]

        # chi4 sidechain dihedrals
        dihedrals['chi4'] = {}
        idx, chi4 = md.compute_chi4(t)
        for i, d in enumerate(idx):
            idx_CA = d[1]
            resid = t.top.atom(idx_CA).residue.index
            dihedrals['chi4'][resid] = chi4[:, i]
        
        return dihedrals
    
    # interface residues only
    elif mode == 'interface':
        dihedrals_interface = {}
        resid_interface = []

        for r in interface_res_rec:
            resid_interface.append(r.index)
        for r in interface_res_lig:
            resid_interface.append(r.index)

        for dihed in dihedrals:
            dihedrals_interface[dihed] = {}
            for resid in dihedrals[dihed]:
                if resid in resid_interface:
                    dihedrals_interface[dihed][resid] = dihedrals[dihed][resid]
        
        return dihedrals_interface
    
    else:
        raise ValueError('Only mode "all" and "interface" implemented.')