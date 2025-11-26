"""
Data utilities for JTNN.
Refactored for Python 3 and MLX compatibility.
"""

import pickle
import os
import random
import numpy as np
import mlx.core as mx
from .moltree import MolTree
from .JTNNEncoder import JTNNEncoder
from .JTNNmpnn import JTMPN

# Note: MPN (Molecular Property Network) is referenced but not implemented
# If needed, it should be added separately or replaced with JTMPN


class PairTreeFolder(object):
    """
    Data folder iterator for paired tree data.
    Yields batches of paired molecular trees.
    """

    def __init__(
        self, data_folder, vocab, batch_size, num_workers=4,
        shuffle=True, y_assm=True, replicate=None
    ):
        """
        Initialize PairTreeFolder.
        
        Args:
            data_folder: Path to folder containing data files
            vocab: Vocabulary object
            batch_size: Batch size for processing
            num_workers: Number of worker processes (not used in MLX version)
            shuffle: Whether to shuffle data
            y_assm: Whether to include assembly information for second batch
            replicate: Number of times to replicate data files
        """
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers  # Kept for compatibility
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        """Iterate over data files and yield batches."""
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:  # Binary mode for pickle
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # Shuffle data before batch

            batches = [
                data[i:i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            # Process batches directly (no DataLoader needed)
            for batch in batches:
                batch0, batch1 = zip(*batch)
                yield (
                    tensorize(batch0, self.vocab, assm=False),
                    tensorize(batch1, self.vocab, assm=self.y_assm)
                )

            del data, batches


class MolTreeFolder(object):
    """
    Data folder iterator for molecular tree data.
    Yields batches of molecular trees.
    """

    def __init__(
        self, data_folder, vocab, batch_size, num_workers=4,
        shuffle=True, assm=True, replicate=None, max_files=None
    ):
        """
        Initialize MolTreeFolder.
        
        Args:
            data_folder: Path to folder containing data files
            vocab: Vocabulary object
            batch_size: Batch size for processing
            num_workers: Number of worker processes (not used in MLX version)
            shuffle: Whether to shuffle data
            assm: Whether to include assembly information
            replicate: Number of times to replicate data files
            max_files: Maximum number of files to load (None = all files)
        """
        self.data_folder = data_folder
        self.data_files = sorted([fn for fn in os.listdir(data_folder) if fn.endswith('.pkl')])
        if max_files is not None:
            self.data_files = self.data_files[:max_files]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers  # Kept for compatibility
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        """Iterate over data files and yield batches."""
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:  # Binary mode for pickle
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # Shuffle data before batch

            # Process batches one at a time to save memory
            # Don't create all batches upfront
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    break  # Skip incomplete batches
                yield tensorize(batch, self.vocab, assm=self.assm)
                # Explicitly delete batch to free memory
                del batch

            # Free data immediately after processing file
            del data


class PairTreeDataset(object):
    """
    Dataset for paired tree data.
    Compatible interface for potential future DataLoader usage.
    """

    def __init__(self, data, vocab, y_assm):
        """
        Initialize PairTreeDataset.
        
        Args:
            data: List of paired data items
            vocab: Vocabulary object
            y_assm: Whether to include assembly information for second batch
        """
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at index."""
        batch0, batch1 = zip(*self.data[idx])
        return (
            tensorize(batch0, self.vocab, assm=False),
            tensorize(batch1, self.vocab, assm=self.y_assm)
        )


class MolTreeDataset(object):
    """
    Dataset for molecular tree data.
    Compatible interface for potential future DataLoader usage.
    """

    def __init__(self, data, vocab, assm=True):
        """
        Initialize MolTreeDataset.
        
        Args:
            data: List of molecular tree batches
            vocab: Vocabulary object
            assm: Whether to include assembly information
        """
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at index."""
        return tensorize(self.data[idx], self.vocab, assm=self.assm)


def tensorize(tree_batch, vocab, assm=True):
    """
    Convert a batch of molecular trees to tensor format.
    
    Args:
        tree_batch: List of MolTree objects
        vocab: Vocabulary object
        assm: Whether to include assembly information
    
    Returns:
        If assm=False: (tree_batch, jtenc _holder, mpn_holder)
        If assm=True: (tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx))
    """
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    
    # Note: MPN is not implemented - using None as placeholder
    # If MPN is needed, it should be implemented separately
    mpn_holder = None  # MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    # Optimize candidate collection with list comprehensions
    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        # Pre-filter nodes to avoid repeated checks
        nodes_with_cands = [node for node in mol_tree.nodes 
                           if not node.is_leaf and len(node.cands) > 1]
        for node in nodes_with_cands:
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    # Handle case where there are no candidates
    if len(cands) == 0:
        # Return empty tensors
        jtmpn_holder = None
        batch_idx = mx.array([], dtype=mx.int32)
    else:
        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
        batch_idx = mx.array(batch_idx, dtype=mx.int32)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    """
    Set node IDs and word IDs for all nodes in a batch.
    Optimized to reduce function call overhead.
    
    Args:
        mol_batch: List of MolTree objects
        vocab: Vocabulary object
    """
    tot = 0
    get_index = vocab.get_index  # Cache method lookup
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = get_index(node.smiles)  # Use cached method
            tot += 1
