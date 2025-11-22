"""
MLX utility functions for JTNN.
Converted from PyTorch to Apple MLX for Apple Silicon acceleration.
"""

import mlx.core as mx
import mlx.nn as nn


def create_var(array, requires_grad=None):
    """
    Create an MLX array (MLX arrays are always differentiable).
    
    Args:
        array: Input array (numpy array or MLX array)
        requires_grad: Ignored (MLX arrays are always differentiable)
    
    Returns:
        MLX array
    """
    if not isinstance(array, mx.array):
        array = mx.array(array)
    return array


def index_select_ND(source, dim, index):
    """
    Select elements from source tensor along dimension using index.
    
    Args:
        source: Source array (MLX array)
        dim: Dimension to select from
        index: Index array (MLX array)
    
    Returns:
        Selected array with reshaped dimensions
    """
    index_shape = index.shape
    suffix_dims = source.shape[1:]
    
    # Flatten index for selection
    flat_index = mx.reshape(index, (-1,))
    
    # Select along the specified dimension using advanced indexing
    if dim == 0:
        target = source[flat_index]
    else:
        # For other dimensions, use take
        target = mx.take(source, flat_index, axis=dim)
    
    # Reshape to final size
    final_shape = index_shape + suffix_dims
    return mx.reshape(target, final_shape)


def avg_pool(all_vecs, scope, dim):
    """
    Average pooling over a dimension using scope information.
    
    Args:
        all_vecs: Input vectors (MLX array)
        scope: List of tuples (start, length) for each batch item
        dim: Dimension to pool over
    
    Returns:
        Averaged vectors
    """
    sizes = mx.array([length for _, length in scope], dtype=mx.float32)
    
    # Sum over the specified dimension
    summed = mx.sum(all_vecs, axis=dim, keepdims=True)
    
    # Divide by sizes, expanding dimensions for proper broadcasting
    # If dim=1 and all_vecs is (batch, seq, features), we need (batch, 1, 1)
    sizes_expanded = mx.reshape(sizes, (-1,) + (1,) * (len(all_vecs.shape) - 1))
    
    return summed / sizes_expanded


def stack_pad_tensor(tensor_list):
    """
    Stack a list of tensors with padding to make them the same length.
    
    Args:
        tensor_list: List of MLX arrays
    
    Returns:
        Stacked array with padding
    """
    max_len = max([t.shape[0] for t in tensor_list])
    
    padded_list = []
    for tensor in tensor_list:
        pad_len = max_len - tensor.shape[0]
        if pad_len > 0:
            # Pad along the first dimension
            # mx.pad takes pad_width as a tuple/list of (before, after) for each dim
            if len(tensor.shape) == 1:
                # 1D: pad at the end
                padded = mx.pad(tensor, (0, pad_len), mode='constant')
            else:
                # 2D+: pad first dimension, no padding on other dims
                pad_width = [(0, pad_len)] + [(0, 0)] * (len(tensor.shape) - 1)
                padded = mx.pad(tensor, pad_width, mode='constant')
            padded_list.append(padded)
        else:
            padded_list.append(tensor)
    
    return mx.stack(padded_list, axis=0)


def flatten_tensor(tensor, scope):
    """
    Convert 3D padded tensor to 2D matrix, removing padded zeros.
    
    Args:
        tensor: 3D tensor (batch, max_len, features)
        scope: List of tuples (start, length) for each batch item
    
    Returns:
        2D matrix with padded zeros removed
    """
    assert tensor.shape[0] == len(scope), "Batch size mismatch"
    
    tlist = []
    for i, tup in enumerate(scope):
        length = tup[1]
        # Extract the actual (non-padded) portion
        tlist.append(tensor[i, 0:length])
    
    return mx.concatenate(tlist, axis=0)


def inflate_tensor(tensor, scope):
    """
    Convert 2D matrix to 3D padded tensor.
    
    Args:
        tensor: 2D tensor (total_length, features)
        scope: List of tuples (start, length) for each batch item
    
    Returns:
        3D padded tensor (batch, max_len, features)
    """
    max_len = max([length for _, length in scope])
    batch_vecs = []
    
    for start, length in scope:
        cur_vecs = tensor[start:start + length]
        
        # Pad to max_len
        pad_len = max_len - length
        if pad_len > 0:
            if len(cur_vecs.shape) == 1:
                # 1D: pad at the end
                cur_vecs = mx.pad(cur_vecs, (0, pad_len), mode='constant')
            else:
                # 2D+: pad first dimension
                pad_width = [(0, pad_len)] + [(0, 0)] * (len(cur_vecs.shape) - 1)
                cur_vecs = mx.pad(cur_vecs, pad_width, mode='constant')
        
        batch_vecs.append(cur_vecs)
    
    return mx.stack(batch_vecs, axis=0)


def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    """
    Gated Recurrent Unit (GRU) computation.
    
    Args:
        x: Input tensor (batch, hidden_size)
        h_nei: Neighbor hidden states (batch, num_neighbors, hidden_size)
        W_z: Update gate weight matrix (Linear layer or callable)
        W_r: Reset gate weight matrix (Linear layer or callable)
        U_r: Reset gate neighbor weight matrix (Linear layer or callable)
        W_h: Hidden state weight matrix (Linear layer or callable)
    
    Returns:
        New hidden state (batch, hidden_size)
    """
    hidden_size = x.shape[-1]
    
    # Sum neighbor hidden states
    sum_h = mx.sum(h_nei, axis=1)  # (batch, hidden_size)
    
    # Update gate
    z_input = mx.concatenate([x, sum_h], axis=1)
    z = mx.sigmoid(W_z(z_input))
    
    # Reset gate
    r_1 = mx.reshape(W_r(x), (-1, 1, hidden_size))
    r_2 = U_r(h_nei)
    r = mx.sigmoid(r_1 + r_2)
    
    # Gated hidden states
    gated_h = r * h_nei
    sum_gated_h = mx.sum(gated_h, axis=1)
    
    # New hidden state
    h_input = mx.concatenate([x, sum_gated_h], axis=1)
    pre_h = mx.tanh(W_h(h_input))
    
    new_h = (1.0 - z) * sum_h + z * pre_h
    
    return new_h
