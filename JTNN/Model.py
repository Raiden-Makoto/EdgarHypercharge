"""
JTNN VAE Model using Apple MLX.
Converted from PyTorch to MLX for Apple Silicon acceleration.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import copy
import rdkit
import rdkit.Chem as Chem
from .moltree import Vocab, MolTree
from .mlxutils import create_var
from .JTNNEncoder import JTNNEncoder
from .JTNNDecoder import JTNNDecoder, cross_entropy_loss
from .JTNNmpnn import JTMPN
from .datautils import tensorize
from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols

# Note: MPN (Molecular Property Network) is not implemented
# Using JTMPN as a placeholder - MPN should be implemented separately if needed


class JTNNVAE(nn.Module):
    """Junction Tree Neural Network Variational Autoencoder."""

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        """
        Initialize JTNN VAE.
        
        Args:
            vocab: Vocabulary object
            hidden_size: Size of hidden representations
            latent_size: Size of latent space (will be divided by 2 for tree/mol)
            depthT: Depth for tree encoder
            depthG: Depth for graph encoder
        """
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        # Tree and Mol have two vectors
        self.latent_size = int(latent_size / 2)

        self.jtnn = JTNNEncoder(
            hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size)
        )
        self.decoder = JTNNDecoder(
            vocab, hidden_size, self.latent_size,
            nn.Embedding(vocab.size(), hidden_size)
        )

        self.jtmpn = JTMPN(hidden_size, depthG)
        # Note: MPN not implemented - using JTMPN as placeholder
        # If MPN is needed, implement it separately
        self.mpn = JTMPN(hidden_size, depthG)  # Placeholder

        self.A_assm = nn.Linear(self.latent_size, hidden_size, bias=False)
        # Loss function will be implemented inline

        self.T_mean = nn.Linear(hidden_size, self.latent_size)
        self.T_var = nn.Linear(hidden_size, self.latent_size)
        self.G_mean = nn.Linear(hidden_size, self.latent_size)
        self.G_var = nn.Linear(hidden_size, self.latent_size)

    def encode(self, jtenc_holder, mpn_holder):
        """
        Encode molecular trees and molecules.
        
        Args:
            jtenc_holder: JTNN encoder inputs
            mpn_holder: MPN inputs (currently None/placeholder)
        
        Returns:
            tree_vecs, tree_mess, mol_vecs
        """
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        # Note: MPN is not implemented, returning zeros as placeholder
        if mpn_holder is None:
            mol_vecs = mx.zeros((tree_vecs.shape[0], self.hidden_size))
        else:
            # If MPN is implemented, use it here
            mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs
    
    def encode_from_smiles(self, smiles_list):
        """
        Encode SMILES strings to latent representations.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            Combined latent vectors (tree + mol)
        """
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(
            tree_batch, self.vocab, assm=False
        )
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return mx.concatenate([tree_vecs, mol_vecs], axis=-1)

    def encode_latent(self, jtenc_holder, mpn_holder):
        """
        Encode to latent space (mean and variance).
        
        Args:
            jtenc_holder: JTNN encoder inputs
            mpn_holder: MPN inputs
        
        Returns:
            mean, var: Latent mean and variance vectors
        """
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        # Note: MPN placeholder
        if mpn_holder is None:
            mol_vecs = mx.zeros((tree_vecs.shape[0], self.hidden_size))
        else:
            mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -mx.abs(self.T_var(tree_vecs))
        mol_var = -mx.abs(self.G_var(mol_vecs))
        return (
            mx.concatenate([tree_mean, mol_mean], axis=1),
            mx.concatenate([tree_var, mol_var], axis=1)
        )

    def rsample(self, z_vecs, W_mean, W_var):
        """
        Reparameterization sampling for VAE.
        
        Args:
            z_vecs: Input vectors
            W_mean: Mean linear layer
            W_var: Variance linear layer
        
        Returns:
            z_vecs: Sampled vectors
            kl_loss: KL divergence loss
        """
        batch_size = z_vecs.shape[0]
        z_mean = W_mean(z_vecs)
        # Following Mueller et al.
        z_log_var = -mx.abs(W_var(z_vecs))
        kl_loss = (
            -0.5 * mx.sum(
                1.0 + z_log_var - z_mean * z_mean - mx.exp(z_log_var)
            ) / batch_size
        )
        epsilon = create_var(
            mx.random.normal(z_mean.shape, dtype=mx.float32)
        )
        z_vecs = z_mean + mx.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        """
        Sample from prior distribution and decode.
        
        Args:
            prob_decode: Whether to use probabilistic decoding
        
        Returns:
            Decoded molecule SMILES
        """
        z_tree = mx.random.normal((1, self.latent_size), dtype=mx.float32)
        z_mol = mx.random.normal((1, self.latent_size), dtype=mx.float32)
        return self.decode(z_tree, z_mol, prob_decode)

    def __call__(self, x_batch, beta):
        """
        Forward pass of the VAE.
        
        Args:
            x_batch: Batch of molecular trees
            beta: KL divergence weight
        
        Returns:
            total_loss, kl_div, word_acc, topo_acc, assm_acc
        """
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(
            x_jtenc_holder, x_mpn_holder
        )
        z_tree_vecs, tree_kl = self.rsample(
            x_tree_vecs, self.T_mean, self.T_var
        )
        z_mol_vecs, mol_kl = self.rsample(
            x_mol_vecs, self.G_mean, self.G_var
        )

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            x_batch, z_tree_vecs
        )
        assm_loss, assm_acc = self.assm(
            x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
        )

        return (
            word_loss + topo_loss + assm_loss + beta * kl_div,
            kl_div.item(), word_acc, topo_acc, assm_acc
        )

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        """
        Assembly loss computation.
        
        Args:
            mol_batch: Batch of molecular trees
            jtmpn_holder: JTMPN inputs
            x_mol_vecs: Molecular vectors
            x_tree_mess: Tree messages
        
        Returns:
            assm_loss, assm_acc
        """
        if jtmpn_holder is None:
            return mx.array(0.0), 0.0
        
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(
            fatoms, fbonds, agraph, bgraph, scope, x_tree_mess
        )

        x_mol_vecs = x_mol_vecs[batch_idx]
        x_mol_vecs = self.A_assm(x_mol_vecs)  # bilinear
        
        # Batch matrix multiplication: (batch, 1, hidden) @ (batch, hidden, 1)
        x_mol_expanded = mx.expand_dims(x_mol_vecs, axis=1)  # (batch, 1, hidden)
        cand_vecs_expanded = mx.expand_dims(cand_vecs, axis=-1)  # (batch, hidden, 1)
        scores_mat = mx.matmul(x_mol_expanded, cand_vecs_expanded)  # (batch, 1, 1)
        scores = mx.squeeze(scores_mat, axis=(1, 2))  # (batch,)
        
        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            # Pre-filter nodes once
            comp_nodes = [node for node in mol_tree.nodes
                         if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                # Cache node.cands to avoid repeated attribute access
                node_cands = node.cands
                label = node_cands.index(node.label)
                ncand = len(node_cands)
                cur_score = scores[tot:tot + ncand]
                tot += ncand

                # Optimize max comparison
                cur_max = mx.max(cur_score)
                if float(cur_score[label].item()) >= float(cur_max.item()):
                    acc += 1

                # Cross entropy loss for single prediction
                label_arr = mx.array([label], dtype=mx.int32)
                cur_score_reshaped = mx.reshape(cur_score, (1, -1))
                loss = cross_entropy_loss(cur_score_reshaped, label_arr)
                all_loss.append(loss)

        if len(all_loss) == 0:
            return mx.array(0.0), 0.0
        
        all_loss_sum = sum(all_loss)
        all_loss_avg = all_loss_sum / len(mol_batch)
        acc_rate = acc * 1.0 / cnt if cnt > 0 else 0.0
        
        return all_loss_avg, acc_rate

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        """
        Decode latent vectors to molecular SMILES.
        
        Args:
            x_tree_vecs: Tree latent vectors (1, latent_size)
            x_mol_vecs: Molecular latent vectors (1, latent_size)
            prob_decode: Whether to use probabilistic decoding
        
        Returns:
            Decoded SMILES string or None
        """
        # Currently do not support batch decoding
        assert x_tree_vecs.shape[0] == 1 and x_mol_vecs.shape[0] == 1

        pred_root, pred_nodes = self.decoder.decode(
            x_tree_vecs, prob_decode
        )
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(
            pred_nodes, scope
        )
        _, tree_mess = self.jtnn(*jtenc_holder)
        # Important: tree_mess is a matrix, mess_dict is a python dict
        tree_mess = (tree_mess, mess_dict)

        x_mol_vecs = mx.squeeze(self.A_assm(x_mol_vecs))  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {
            atom.GetIdx(): atom.GetIdx()
            for atom in cur_mol.GetAtoms()
        }

        cur_mol, _ = self.dfs_assemble(
            tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [],
            pred_root, None, prob_decode, check_aroma=True
        )
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {
                atom.GetIdx(): atom.GetIdx()
                for atom in cur_mol.GetAtoms()
            }
            cur_mol, pre_mol = self.dfs_assemble(
                tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [],
                pred_root, None, prob_decode, check_aroma=False
            )
            if cur_mol is None:
                cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return (
            Chem.MolToSmiles(cur_mol)
            if cur_mol is not None else None
        )
        
    def dfs_assemble(
        self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap,
        fa_amap, cur_node, fa_node, prob_decode, check_aroma
    ):
        """
        Depth-first search assembly of molecular tree.
        
        Args:
            y_tree_mess: Tree messages tuple (matrix, dict)
            x_mol_vecs: Molecular vectors
            all_nodes: All nodes in the tree
            cur_mol: Current molecule being assembled
            global_amap: Global atom mapping
            fa_amap: Father atom mapping
            cur_node: Current node
            fa_node: Father node
            prob_decode: Whether to use probabilistic decoding
            check_aroma: Whether to check aromaticity
        
        Returns:
            cur_mol, pre_mol: Assembled molecule and previous molecule
        """
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [
            nei for nei in cur_node.neighbors if nei.nid != fa_nid
        ]
        neighbors = [
            nei for nei in children if nei.mol.GetNumAtoms() > 1
        ]
        neighbors = sorted(
            neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True
        )
        singletons = [
            nei for nei in children if nei.mol.GetNumAtoms() == 1
        ]
        neighbors = singletons + neighbors

        cur_amap = [
            (fa_nid, a2, a1) for nid, a1, a2 in fa_amap
            if nid == cur_node.nid
        ]
        cands, aroma_score = enum_assemble(
            cur_node, neighbors, prev_nodes, cur_amap
        )
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)
        aroma_score = mx.array(aroma_score, dtype=mx.float32)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(
                fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0]
            )
            # Matrix-vector multiplication: (num_cands, hidden) @ (hidden,)
            scores = mx.matmul(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = mx.array([1.0], dtype=mx.float32)

        if prob_decode:
            probs = mx.softmax(mx.reshape(scores, (1, -1)), axis=1)
            probs = mx.squeeze(probs) + 1e-7  # Prevent prob = 0
            # Normalize to ensure sum = 1
            probs = probs / mx.sum(probs)
            # Sample from distribution using cumulative sum
            probs_cumsum = mx.cumsum(probs)
            uniform = mx.random.uniform((1,)).item()
            # Find first index where cumsum >= uniform
            # Use searchsorted-like approach
            mask = (probs_cumsum >= uniform)
            cand_idx_val = mx.argmax(mask.astype(mx.int32)).item()
            # Ensure we get a valid index
            if not mask[cand_idx_val].item() and cand_idx_val == 0:
                cand_idx_val = len(probs) - 1
            cand_idx = mx.array([cand_idx_val], dtype=mx.int32)
        else:
            # Get indices in descending order
            cand_idx = mx.argsort(scores)[::-1]  # Descending

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.size):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = (
                    new_global_amap[cur_node.nid][ctr_atom]
                )

            # Father is already attached
            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess, x_mol_vecs, all_nodes, cur_mol,
                    new_global_amap, pred_amap, nei_node, cur_node,
                    prob_decode, check_aroma
                )
                if tmp_mol is None: 
                    has_error = True
                    if i == 0:
                        pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error:
                return cur_mol, cur_mol

        return None, pre_mol
