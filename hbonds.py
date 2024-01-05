import itertools


def find_hbond_triplets(rec_donors, rec_acceptors, lig_donors, lig_acceptors):
    '''
    Find triplets of atom indices for possible hbond interactions between receptor and ligand.

    Parameters
    ----------
    rec_donors : list of Atom
        Receptor Donor atoms.
    rec_acceptors : list of Atom
        Receptor Acceptor atoms.
    lig_donors : list of Atom
        Ligand Donor atoms.
    lig_acceptors : list of Atom
        Ligand Acceptor atoms.

    Returns
    -------
    triplets: list of tuple of int
        Hbond triplets in format (D, H, A).
    '''
    rec_acceptor_idx = [x.index for x in rec_acceptors]
    lig_acceptor_idx = [x for x in lig_acceptors]

    rec_donor_duplets = []
    for a in rec_donors:
        h_atoms = []
        for b in a.bonds:
            if b[1] == 'H':
                h_atoms.append(b[0])
        for x in h_atoms:
            rec_donor_duplets.append([a.index, x])

    lig_donor_duplets = []
    for a in lig_donors:
        h_atoms = []
        for b in a.bonds:
            if b[1] == 'H':
                h_atoms.append(b[0])
        for x in h_atoms:
            lig_donor_duplets.append([a.index, x])
    
    triplets = [(x[0][0], x[0][1], x[1]) for x in itertools.product(rec_donor_duplets, lig_acceptor_idx)] + [(x[0][0], x[0][1], x[1]) for x in itertools.product(lig_donor_duplets, rec_acceptor_idx)]
    return triplets