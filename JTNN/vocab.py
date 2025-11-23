# from https://github.com/wengong-jin/icml18-jtnn/blob/master/fast_jtnn/vocab.py

import copy

import rdkit.Chem as Chem


def get_slots(smiles):
    """Extract atom slots (symbol, formal charge, total H count) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return [
        (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
        for atom in mol.GetAtoms()
    ]


class Vocab(object):
    """Vocabulary class for managing SMILES strings and their atom slots."""
    
    benzynes = [
        'C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1',
        'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1',
        'C1=NN=CN=N1'
    ]
    
    penzynes = [
        'C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1',
        'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1',
        'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1',
        'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1', 'C1=NN=N[NH]1',
        'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1'
    ]

    def __init__(self, smiles_list):
        """Initialize vocabulary with a list of SMILES strings."""
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        
        # Update benzynes: 6-atom rings with >= 2 double bonds
        Vocab.benzynes = [
            s for s in smiles_list
            if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6
        ] + ['C1=CCNCC1']
        
        # Update penzynes: 5-atom rings with >= 2 double bonds
        Vocab.penzynes = [
            s for s in smiles_list
            if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5
        ] + ['C1=NCCN1', 'C1=NNCC1']
        
    def get_index(self, smiles):
        """Get the index of a SMILES string in the vocabulary."""
        return self.vmap[smiles]

    def get_smiles(self, idx):
        """Get the SMILES string at the given index."""
        return self.vocab[idx]

    def get_slots(self, idx):
        """Get a deep copy of the atom slots at the given index."""
        return copy.deepcopy(self.slots[idx])

    def size(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)
