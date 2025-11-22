"""
JTNN Message Passing Neural Network using Apple MLX.
Converted from PyTorch to MLX for Apple Silicon acceleration.
"""

import mlx.core as mx
import mlx.nn as nn
from .mlxutils import create_var, index_select_ND
from .chemutils import get_mol
import rdkit.Chem as Chem

ELEM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown'
]

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5
MAX_NB = 15


def onek_encoding_unk(x, allowable_set):
    """
    One-hot encoding with unknown fallback.
    
    Args:
        x: Value to encode
        allowable_set: List of allowable values
    
    Returns:
        List of boolean values (one-hot encoding)
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [s == x for s in allowable_set]


def atom_features(atom):
    """
    Extract atom features as a tensor.
    
    Args:
        atom: RDKit atom object
    
    Returns:
        MLX array of atom features
    """
    features = (
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        [atom.GetIsAromatic()]
    )
    return mx.array(features, dtype=mx.float32)


def bond_features(bond):
    """
    Extract bond features as a tensor.
    
    Args:
        bond: RDKit bond object
    
    Returns:
        MLX array of bond features
    """
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing()
    ]
    return mx.array(features, dtype=mx.float32)


class JTMPN(nn.Module):
    """Junction Tree Message Passing Network."""

    def __init__(self, hidden_size, depth):
        """
        Initialize JTMPN.
        
        Args:
            hidden_size: Size of hidden representations
            depth: Depth of message passing iterations
        """
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def __call__(self, fatoms, fbonds, agraph, bgraph, scope, tree_message):
        """
        Forward pass of JTMPN.
        
        Args:
            fatoms: Atom features (num_atoms, ATOM_FDIM)
            fbonds: Bond features (num_bonds, ATOM_FDIM + BOND_FDIM)
            agraph: Atom graph structure (num_atoms, MAX_NB)
            bgraph: Bond graph structure (num_bonds, MAX_NB)
            scope: List of (start, length) tuples for each molecule
            tree_message: Tree message vectors (num_messages, hidden_size)
                          tree_message[0] should be vec(0) for padding
        
        Returns:
            mol_vecs: Molecule-level representations (batch, hidden_size)
        """
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        graph_message = mx.maximum(
            mx.zeros_like(binput), binput
        )  # ReLU

        for i in range(self.depth - 1):
            message = mx.concatenate([tree_message, graph_message], axis=0)
            nei_message = index_select_ND(message, 0, bgraph)
            # Assuming tree_message[0] == vec(0)
            nei_message = mx.sum(nei_message, axis=1)
            nei_message = self.W_h(nei_message)
            graph_message = mx.maximum(
                mx.zeros_like(binput + nei_message),
                binput + nei_message
            )  # ReLU

        message = mx.concatenate([tree_message, graph_message], axis=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = mx.sum(nei_message, axis=1)
        ainput = mx.concatenate([fatoms, nei_message], axis=1)
        atom_hiddens = mx.maximum(
            mx.zeros_like(self.W_o(ainput)),
            self.W_o(ainput)
        )  # ReLU

        mol_vecs = []
        for st, le in scope:
            # Extract slice and average
            mol_vec = mx.sum(atom_hiddens[st:st + le], axis=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = mx.stack(mol_vecs, axis=0)
        return mol_vecs

    @staticmethod
    def tensorize(cand_batch, mess_dict):
        """
        Convert candidate batch to tensor format.
        
        Args:
            cand_batch: List of (smiles, all_nodes, ctr_node) tuples
            mess_dict: Dictionary mapping (node_idx, node_idx) to message index
        
        Returns:
            Tuple of (fatoms, fbonds, agraph, bgraph, scope)
        """
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        total_atoms = 0
        total_mess = len(mess_dict) + 1  # Must include vec(0) padding
        scope = []

        for smiles, all_nodes, ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            # The original jtnn version kekulizes. Need to revisit why it is necessary
            Chem.Kekulize(mol)
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                # Here x_nid, y_nid could be 0
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = total_mess + len(all_bonds)  # Bond idx offset by total_mess
                all_bonds.append((x, y))
                fbonds.append(mx.concatenate([fatoms[x], bfeature], axis=0))
                in_bonds[y].append(b)

                b = total_mess + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(mx.concatenate([fatoms[y], bfeature], axis=0))
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms = mx.stack(fatoms, axis=0)
        fbonds = mx.stack(fbonds, axis=0)

        # Build agraph and bgraph using list comprehension then convert
        agraph_list = []
        for a in range(total_atoms):
            row = in_bonds[a][:MAX_NB]
            row.extend([0] * (MAX_NB - len(row)))
            agraph_list.append(row)

        bgraph_list = []
        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            row = []
            for b2 in in_bonds[x]:  # b2 is offset by total_mess
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    row.append(b2)
                    if len(row) >= MAX_NB:
                        break
            row.extend([0] * (MAX_NB - len(row)))
            bgraph_list.append(row)

        agraph = mx.array(agraph_list, dtype=mx.int32)
        bgraph = mx.array(bgraph_list, dtype=mx.int32)

        return (fatoms, fbonds, agraph, bgraph, scope)
