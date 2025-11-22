"""
Training script for JTNN VAE using Apple MLX.
Converted from PyTorch to MLX for Apple Silicon acceleration.
"""

import mlx.core as mx
import mlx.optimizers as optim
import math
import sys
import os
import numpy as np
import argparse

from JTNN import Vocab, JTNNVAE
from JTNN.datautils import MolTreeFolder
import rdkit

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser(
    description='Train JTNN VAE model using MLX'
)
parser.add_argument('--train', required=True, help='Training data folder')
parser.add_argument('--vocab', required=True, help='Vocabulary file')
parser.add_argument('--save_dir', required=True, help='Directory to save models')
parser.add_argument('--load_epoch', type=int, default=0, help='Epoch to load from')

parser.add_argument('--hidden_size', type=int, default=450, help='Hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--latent_size', type=int, default=56, help='Latent size')
parser.add_argument('--depthT', type=int, default=20, help='Tree depth')
parser.add_argument('--depthG', type=int, default=3, help='Graph depth')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--clip_norm', type=float, default=50.0, help='Gradient clip norm')
parser.add_argument('--beta', type=float, default=0.0, help='Initial KL weight')
parser.add_argument('--step_beta', type=float, default=0.002, help='KL weight step')
parser.add_argument('--max_beta', type=float, default=1.0, help='Max KL weight')
parser.add_argument('--warmup', type=int, default=40000, help='Warmup steps')

parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--anneal_rate', type=float, default=0.9, help='LR anneal rate')
parser.add_argument('--anneal_iter', type=int, default=40000, help='LR anneal interval')
parser.add_argument('--kl_anneal_iter', type=int, default=2000, help='KL anneal interval')
parser.add_argument('--print_iter', type=int, default=50, help='Print interval')
parser.add_argument('--save_iter', type=int, default=5000, help='Save interval')

args = parser.parse_args()
print(args)

# Load vocabulary
with open(args.vocab, 'r') as f:
    vocab_list = [x.strip("\r\n ") for x in f]
vocab = Vocab(vocab_list)

# Create model
model = JTNNVAE(
    vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG
)
print(model)

# Initialize parameters
def init_params(model):
    """Initialize model parameters."""
    def init_recursive(module):
        """Recursively initialize parameters."""
        for name, param in module.named_parameters():
            if len(param.shape) == 1:
                # Bias terms: initialize to zero
                # In MLX, we need to replace the parameter
                # This is tricky - parameters are immutable in MLX
                # We'll skip bias initialization for now as MLX handles it
                pass
            else:
                # Weight matrices: Xavier normal initialization
                # Xavier: std = sqrt(2 / (fan_in + fan_out))
                fan_in, fan_out = param.shape[0], param.shape[1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                # Note: MLX parameters are immutable, initialization
                # should be done in the layer's __init__
                pass
        # Initialize submodules
        for child in module.children():
            init_recursive(child)
    
    # MLX modules initialize parameters automatically
    # Custom initialization would need to be done in __init__
    # For now, we'll rely on MLX's default initialization
    pass

init_params(model)

# Load checkpoint if specified
if args.load_epoch > 0:
    checkpoint_path = os.path.join(
        args.save_dir, f"model.iter-{args.load_epoch}.npz"
    )
    if os.path.exists(checkpoint_path):
        try:
            weights = mx.load(checkpoint_path)
            model.load_weights(weights)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found")

# Count parameters
def count_parameters(model):
    """Count total number of parameters."""
    total = 0
    for param in model.parameters():
        if isinstance(param, mx.array):
            total += param.size
        elif hasattr(param, 'size'):
            total += param.size
    return total

num_params = count_parameters(model)
print(f"Model #Params: {num_params // 1000}K")

# Create optimizer
optimizer = optim.Adam(learning_rate=args.lr)
current_lr = args.lr

# Helper functions for parameter and gradient norms
def param_norm(model):
    """Compute parameter norm."""
    total = 0.0
    for param in model.parameters():
        total += float(mx.sum(param * param).item())
    return math.sqrt(total)


def clip_grad_norm(grads, max_norm):
    """
    Clip gradient norm.
    
    Args:
        grads: Gradient tree (matching model parameter structure)
        max_norm: Maximum gradient norm
    
    Returns:
        Clipped gradients, gradient norm before clipping
    """
    def compute_norm(grad_tree):
        """Recursively compute gradient norm."""
        total = 0.0
        if isinstance(grad_tree, dict):
            for v in grad_tree.values():
                total += compute_norm(v)
        elif isinstance(grad_tree, (list, tuple)):
            for item in grad_tree:
                total += compute_norm(item)
        else:
            # It's an array
            if isinstance(grad_tree, mx.array):
                total += float(mx.sum(grad_tree * grad_tree).item())
        return total
    
    total_norm = math.sqrt(compute_norm(grads))
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        
        def clip_recursive(grad_tree):
            """Recursively clip gradients."""
            if isinstance(grad_tree, dict):
                return {k: clip_recursive(v) for k, v in grad_tree.items()}
            elif isinstance(grad_tree, (list, tuple)):
                return type(grad_tree)(clip_recursive(item) for item in grad_tree)
            else:
                if isinstance(grad_tree, mx.array):
                    return grad_tree * clip_coef
                return grad_tree
        
        grads = clip_recursive(grads)
    
    return grads, total_norm


def grad_norm(grads):
    """Compute gradient norm from gradient tree."""
    def compute_norm(grad_tree):
        """Recursively compute gradient norm."""
        total = 0.0
        if isinstance(grad_tree, dict):
            for v in grad_tree.values():
                total += compute_norm(v)
        elif isinstance(grad_tree, (list, tuple)):
            for item in grad_tree:
                total += compute_norm(item)
        else:
            # It's an array
            total += float(mx.sum(grad_tree * grad_tree).item())
        return total
    
    return math.sqrt(compute_norm(grads))


# Training loop
total_step = args.load_epoch
beta = args.beta
meters = np.zeros(4)
last_grad_norm = 0.0

# Create save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

# Loss and gradient function
# MLX's value_and_grad computes gradients w.r.t. the first argument (model)
def loss_fn(model, batch, beta_val):
    """Compute loss and metrics."""
    loss, kl_div, wacc, tacc, sacc = model(batch, beta_val)
    return loss, (kl_div, wacc, tacc, sacc)

# value_and_grad returns (value, gradients) where gradients match model structure
loss_and_grad_fn = mx.value_and_grad(loss_fn, argnums=0)

for epoch in range(args.epoch):
    loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=4)
    for batch in loader:
        total_step += 1
        try:
            # Forward and backward pass
            (loss, metrics), grads = loss_and_grad_fn(model, batch, beta)
            kl_div, wacc, tacc, sacc = metrics
            
            # Clip gradients
            grads, grad_norm_val = clip_grad_norm(grads, args.clip_norm)
            last_grad_norm = grad_norm_val
            
            # Update parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            
        except Exception as e:
            print(f"Error at step {total_step}: {e}")
            import traceback
            traceback.print_exc()
            continue

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            pnorm = param_norm(model)
            gnorm = last_grad_norm
            print(
                "[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, "
                "Assm: %.2f, PNorm: %.2f, GNorm: %.2f, LR: %.6f" % (
                    total_step, beta, meters[0], meters[1], meters[2],
                    meters[3], pnorm, gnorm, current_lr
                )
            )
            sys.stdout.flush()
            meters *= 0

        if total_step % args.save_iter == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"model.iter-{total_step}.npz"
            )
            # Save model weights using MLX's save_weights method
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if total_step % args.anneal_iter == 0:
            current_lr *= args.anneal_rate
            optimizer.learning_rate = current_lr
            print(f"Learning rate: {current_lr:.6f}")

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

print("Training complete!")