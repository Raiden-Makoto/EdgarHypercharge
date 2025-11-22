"""
JTNN Encoder using Apple MLX.
Converted from PyTorch to MLX for Apple Silicon acceleration.
"""

from collections import deque

import mlx.core as mx
import mlx.nn as nn

from .mlxutils import create_var, index_select_ND


class JTNNEncoder(nn.Module):
    """Junction Tree Neural Network Encoder."""

    def __init__(self, hidden_size, depth, embedding):
        """
        Initialize JTNN Encoder.
        
        Args:
            hidden_size: Size of hidden representations
            depth: Depth of GRU iterations
            embedding: Embedding layer for node features
        """
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def __call__(self, fnode, fmess, node_graph, mess_graph, scope):
        """
        Forward pass of the encoder.
        
        Args:
            fnode: Node features
            fmess: Message source indices
            node_graph: Graph structure for nodes
            mess_graph: Graph structure for messages
            scope: List of (start, length) tuples for each tree in batch
        
        Returns:
            tree_vecs: Tree-level representations
            messages: Message representations
        """
        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)
        messages = create_var(
            mx.zeros((mess_graph.shape[0], self.hidden_size))
        )

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = mx.concatenate(
            [fnode, mx.sum(mess_nei, axis=1)], axis=-1
        )
        node_vecs = self.outputNN(node_vecs)

        batch_vecs = []
        for st, le in scope:
            # Root is the first node
            cur_vecs = node_vecs[st]
            batch_vecs.append(cur_vecs)

        tree_vecs = mx.stack(batch_vecs, axis=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        """
        Convert a batch of MolTree objects to tensor format.
        
        Args:
            tree_batch: List of MolTree objects
        
        Returns:
            Tuple of (fnode, fmess, node_graph, mess_graph, scope), mess_dict
        """
        node_batch = []
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)

    @staticmethod
    def tensorize_nodes(node_batch, scope):
        """
        Convert node batch to tensor format.
        
        Args:
            node_batch: List of MolTreeNode objects
            scope: List of (start, length) tuples
        
        Returns:
            Tuple of (fnode, fmess, node_graph, mess_graph, scope), mess_dict
        """
        messages, mess_dict = [None], {}
        fnode = []
        for x in node_batch:
            # x.wid should be set elsewhere (word ID from vocabulary)
            fnode.append(x.wid)
            for y in x.neighbors:
                # x.idx should be the index in node_batch
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx:
                    continue
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        # Convert to MLX arrays (int32 for indices)
        mess_graph = mx.array(mess_graph, dtype=mx.int32)
        node_graph = mx.array(node_graph, dtype=mx.int32)
        fmess = mx.array(fmess, dtype=mx.int32)
        fnode = mx.array(fnode, dtype=mx.int32)
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict


class GraphGRU(nn.Module):
    """Graph GRU for message passing."""

    def __init__(self, input_size, hidden_size, depth):
        """
        Initialize Graph GRU.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            depth: Number of GRU iterations
        """
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def __call__(self, h, x, mess_graph):
        """
        Forward pass of Graph GRU.
        
        Args:
            h: Hidden states (batch, hidden_size)
            x: Input features (batch, input_size)
            mess_graph: Message graph structure
        
        Returns:
            Updated hidden states
        """
        # Create mask: first vector is padding (set to 0)
        mask = mx.ones((h.shape[0], 1), dtype=mx.float32)
        # Set first element to 0 (padding)
        mask = mx.concatenate([mx.zeros((1, 1)), mask[1:]], axis=0)
        mask = create_var(mask)

        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = mx.sum(h_nei, axis=1)
            z_input = mx.concatenate([x, sum_h], axis=1)
            z = mx.sigmoid(self.W_z(z_input))

            r_1 = mx.reshape(
                self.W_r(x), (-1, 1, self.hidden_size)
            )
            r_2 = self.U_r(h_nei)
            r = mx.sigmoid(r_1 + r_2)

            gated_h = r * h_nei
            sum_gated_h = mx.sum(gated_h, axis=1)
            h_input = mx.concatenate([x, sum_gated_h], axis=1)
            pre_h = mx.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h
