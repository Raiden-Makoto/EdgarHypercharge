"""
JTNN Decoder using Apple MLX.
Converted from PyTorch to MLX for Apple Silicon acceleration.
"""

import mlx.core as mx
import mlx.nn as nn
from .moltree import Vocab, MolTree, MolTreeNode
from .mlxutils import create_var, GRU
from .chemutils import enum_assemble, set_atommap

MAX_NB = 15
MAX_DECODE_LEN = 100


def cross_entropy_loss(logits, targets):
    """Cross entropy loss implementation for MLX."""
    # logits: (batch, num_classes)
    # targets: (batch,) with class indices
    batch_size = logits.shape[0]
    log_probs = nn.log_softmax(logits, axis=1)
    # Gather log probabilities for target classes
    batch_indices = mx.arange(batch_size)
    target_log_probs = log_probs[batch_indices, targets]
    return -mx.sum(target_log_probs)


def bce_with_logits_loss(logits, targets):
    """Binary cross entropy with logits loss for MLX."""
    # logits: (batch,)
    # targets: (batch,) with 0 or 1
    # BCE with logits: -sum(target * log(sigmoid(logit)) + (1-target) * log(1-sigmoid(logit)))
    # = sum(log(1 + exp(-logit))) - target * logit
    # More numerically stable: use max(0, logit) - logit * target + log(1 + exp(-abs(logit)))
    max_logit = mx.maximum(mx.zeros_like(logits), logits)
    loss = max_logit - logits * targets + mx.log(1 + mx.exp(-mx.abs(logits)))
    return mx.sum(loss)


class JTNNDecoder(nn.Module):
    """Junction Tree Neural Network Decoder."""

    def __init__(self, vocab, hidden_size, latent_size, embedding):
        """
        Initialize JTNN Decoder.
        
        Args:
            vocab: Vocabulary object
            hidden_size: Size of hidden representations
            latent_size: Size of latent/tree vectors
            embedding: Embedding layer for node features
        """
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding

        # GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Word Prediction Weights
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        # Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        # Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        """
        Aggregate hidden states with tree context vectors.
        
        Args:
            hiddens: Hidden states (batch, hidden_size)
            contexts: Context indices (batch,)
            x_tree_vecs: Tree vectors (num_trees, latent_size)
            mode: 'word' or 'stop'
        
        Returns:
            Output scores
        """
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        # Select tree contexts using indexing
        tree_contexts = x_tree_vecs[contexts]
        input_vec = mx.concatenate([hiddens, tree_contexts], axis=-1)
        output_vec = mx.maximum(mx.zeros_like(V(input_vec)), V(input_vec))  # ReLU
        return V_o(output_vec)

    def __call__(self, mol_batch, x_tree_vecs):
        """
        Forward pass of the decoder.
        
        Args:
            mol_batch: List of MolTree objects
            x_tree_vecs: Tree-level representations (batch, latent_size)
        
        Returns:
            pred_loss, stop_loss, pred_acc, stop_acc
        """
        pred_hiddens, pred_contexts, pred_targets = [], [], []
        stop_hiddens, stop_contexts, stop_targets = [], [], []
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        # Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(
            create_var(mx.zeros((len(mol_batch), self.hidden_size)))
        )
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append(
            create_var(mx.array(list(range(batch_size)), dtype=mx.int32))
        )

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(mx.zeros(self.hidden_size), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in prop_list:
                # Neighbors for message passing (target not included)
                cur_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                    if node_y.idx != real_y.idx
                ]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                # Neighbors for stop prediction (all neighbors)
                cur_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                ]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                # Current clique embedding
                cur_x.append(node_x.wid)

            # Clique embedding
            cur_x = create_var(mx.array(cur_x, dtype=mx.int32))
            cur_x = self.embedding(cur_x) 
            
            # Message passing
            cur_h_nei = mx.stack(cur_h_nei, axis=0)
            cur_h_nei = mx.reshape(
                cur_h_nei, (-1, MAX_NB, self.hidden_size)
            )
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # Node Aggregate
            cur_o_nei = mx.stack(cur_o_nei, axis=0)
            cur_o_nei = mx.reshape(
                cur_o_nei, (-1, MAX_NB, self.hidden_size)
            )
            cur_o = mx.sum(cur_o_nei, axis=1)

            # Gather targets
            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            # Hidden states for stop prediction
            cur_batch = create_var(mx.array(batch_list, dtype=mx.int32))
            stop_hidden = mx.concatenate([cur_x, cur_o], axis=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)
            
            # Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(mx.array(batch_list, dtype=mx.int32))
                pred_contexts.append(cur_batch)

                cur_pred = create_var(mx.array(pred_list, dtype=mx.int32))
                pred_hiddens.append(new_h[cur_pred])
                pred_targets.extend(pred_target)

        # Last stop at root
        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [
                h[(node_y.idx, node_x.idx)]
                for node_y in node_x.neighbors
            ]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(mx.array(cur_x, dtype=mx.int32))
        cur_x = self.embedding(cur_x) 
        cur_o_nei = mx.stack(cur_o_nei, axis=0)
        cur_o_nei = mx.reshape(cur_o_nei, (-1, MAX_NB, self.hidden_size))
        cur_o = mx.sum(cur_o_nei, axis=1)

        stop_hidden = mx.concatenate([cur_x, cur_o], axis=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(
            create_var(mx.array(list(range(batch_size)), dtype=mx.int32))
        )
        stop_targets.extend([0] * len(mol_batch))

        # Predict next clique
        pred_contexts = mx.concatenate(pred_contexts, axis=0)
        pred_hiddens = mx.concatenate(pred_hiddens, axis=0)
        pred_scores = self.aggregate(
            pred_hiddens, pred_contexts, x_tree_vecs, 'word'
        )
        pred_targets = create_var(mx.array(pred_targets, dtype=mx.int32))

        pred_loss = cross_entropy_loss(pred_scores, pred_targets) / len(mol_batch)
        preds = mx.argmax(pred_scores, axis=1)
        pred_acc = mx.equal(preds, pred_targets).astype(mx.float32)
        pred_acc = mx.sum(pred_acc) / pred_targets.size

        # Predict stop
        stop_contexts = mx.concatenate(stop_contexts, axis=0)
        stop_hiddens = mx.concatenate(stop_hiddens, axis=0)
        stop_hiddens = mx.maximum(
            mx.zeros_like(self.U_i(stop_hiddens)),
            self.U_i(stop_hiddens)
        )  # ReLU
        stop_scores = self.aggregate(
            stop_hiddens, stop_contexts, x_tree_vecs, 'stop'
        )
        stop_scores = mx.squeeze(stop_scores, axis=-1)
        stop_targets = create_var(mx.array(stop_targets, dtype=mx.float32))
        
        stop_loss = bce_with_logits_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = mx.greater_equal(stop_scores, mx.zeros_like(stop_scores)).astype(mx.float32)
        stop_acc = mx.equal(stops, stop_targets).astype(mx.float32)
        stop_acc = mx.sum(stop_acc) / stop_targets.size

        return (
            pred_loss.item(), stop_loss.item(),
            pred_acc.item(), stop_acc.item()
        )
    
    def decode(self, x_tree_vecs, prob_decode):
        """
        Decode a molecular tree from tree vectors.
        
        Args:
            x_tree_vecs: Tree-level representation (1, latent_size)
            prob_decode: Whether to use probabilistic decoding
        
        Returns:
            root: Root node of decoded tree
            all_nodes: List of all nodes in decoded tree
        """
        assert x_tree_vecs.shape[0] == 1

        stack = []
        init_hiddens = create_var(mx.zeros((1, self.hidden_size)))
        zero_pad = create_var(mx.zeros((1, 1, self.hidden_size)))
        contexts = create_var(mx.zeros((1,), dtype=mx.int32))

        # Root Prediction
        root_score = self.aggregate(
            init_hiddens, contexts, x_tree_vecs, 'word'
        )
        root_wid = mx.argmax(root_score, axis=1).item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = [
                h[(node_y.idx, node_x.idx)]
                for node_y in node_x.neighbors
            ]
            if len(cur_h_nei) > 0:
                cur_h_nei = mx.stack(cur_h_nei, axis=0)
                cur_h_nei = mx.reshape(cur_h_nei, (1, -1, self.hidden_size))
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(mx.array([node_x.wid], dtype=mx.int32))
            cur_x = self.embedding(cur_x)

            # Predict stop
            cur_h = mx.sum(cur_h_nei, axis=1)
            stop_hiddens = mx.concatenate([cur_x, cur_h], axis=1)
            stop_hiddens = mx.maximum(
                mx.zeros_like(self.U_i(stop_hiddens)),
                self.U_i(stop_hiddens)
            )  # ReLU
            stop_score = self.aggregate(
                stop_hiddens, contexts, x_tree_vecs, 'stop'
            )
            
            if prob_decode:
                # Sample from Bernoulli distribution
                prob = mx.sigmoid(stop_score)
                sample = mx.random.bernoulli(prob)
                backtrack = (sample.item() == 0)
            else:
                backtrack = (stop_score.item() < 0) 

            if not backtrack:  # Forward: Predict next clique
                new_h = GRU(
                    cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h
                )
                pred_score = self.aggregate(
                    new_h, contexts, x_tree_vecs, 'word'
                )

                if prob_decode:
                    # Sample from softmax distribution
                    probs = mx.softmax(pred_score, axis=1)
                    probs_flat = mx.squeeze(probs)
                    # Sample top 5 (get indices in descending order)
                    sort_wid = mx.argsort(probs_flat)[::-1][:5]
                else:
                    # Get indices in descending order
                    sort_wid = mx.argsort(mx.squeeze(pred_score))[::-1][:5]

                next_wid = None
                for wid in sort_wid:
                    wid = wid.item()
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True  # No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:  # Backtrack, use if instead of else
                if len(stack) == 1: 
                    break  # At root, terminate

                node_fa, _ = stack[-2]
                cur_h_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                    if node_y.idx != node_fa.idx
                ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = mx.stack(cur_h_nei, axis=0)
                    cur_h_nei = mx.reshape(cur_h_nei, (1, -1, self.hidden_size))
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(
                    cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h
                )
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes


"""
Helper Functions:
"""


def dfs(stack, x, fa_idx):
    """Depth-first search to build trace for training."""
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    """
    Check if parent and child slots are compatible.
    
    Args:
        fa_slots: List of parent atom slots (symbol, charge, H_count)
        ch_slots: List of child atom slots (symbol, charge, H_count)
    
    Returns:
        True if compatible, False otherwise
    """
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0:
        return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:
        # Never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:
        # Never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    """
    Check if two nodes can be assembled together.
    
    Args:
        node_x: Parent node
        node_y: Child node
    
    Returns:
        True if they can be assembled, False otherwise
    """
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(
        neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True
    )
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands, aroma_scores = enum_assemble(node_x, neighbors)
    return len(cands) > 0  # and sum(aroma_scores) >= 0
