

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