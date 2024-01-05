from utils import RESIDUE_CATIONS, RESIDUE_ANIONS

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


def store_residue(t, resid):
    # basic residue information
    r = t.top.residue(resid)
    R = Residue()
    R.index = resid
    R.name = r.name
    R.resSeq = r.resSeq

    # all bond that involve residue atoms
    bonds = []
    for b in t.top.bonds:
        if b[0].residue == r or b[1].residue == r:
            bonds.append([b[0].index, b[1].index])
    R.bonds = bonds

    # handle missing bond information
    if R.bonds == []:
        print(f'No bond information for {R.name}{R.resSeq}.')

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
        for b in t.top.bonds:
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
        if R.name in RESIDUE_ANIONS:
            if a.name in RESIDUE_ANIONS[R.name]:
                A.is_anion = True
            else:
                A.is_anion = False
        else:
            A.is_anion = False

        # check list of amino acid cation atoms (based on residue name)
        if R.name in RESIDUE_CATIONS:
            if a.name in RESIDUE_CATIONS[R.name]:
                A.is_cation = True
            else:
                A.is_cation = False
        else:
            A.is_cation = False

        atoms.append(A)

    R.atoms = atoms

    return R