"""
Inference script for JTNN VAE using Apple MLX.
Generates molecular samples from the trained model.
"""

import mlx.core as mx
import sys
import argparse
import numpy as np
from JTNN import Vocab, JTNNVAE
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser(
    description='Generate molecular samples using trained JTNN VAE model'
)
parser.add_argument(
    '--nsample', type=int, required=True,
    help='Number of samples to generate'
)
parser.add_argument(
    '--vocab', required=True,
    help='Vocabulary file path'
)
parser.add_argument(
    '--model', required=True,
    help='Path to trained model checkpoint (.npz file)'
)

parser.add_argument(
    '--hidden_size', type=int, default=32,
    help='Hidden size (default: 32 for ~100K params, must match training)'
)
parser.add_argument(
    '--latent_size', type=int, default=8,
    help='Latent size (default: 8 for ~100K params, must match training)'
)
parser.add_argument(
    '--depthT', type=int, default=3,
    help='Tree depth (default: 3 for ~100K params, must match training)'
)
parser.add_argument(
    '--depthG', type=int, default=2,
    help='Graph depth (default: 2 for ~100K params, must match training)'
)
parser.add_argument(
    '--prob_decode', action='store_true',
    help='Use probabilistic decoding instead of greedy'
)
parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed for reproducibility'
)

args = parser.parse_args()
   
# Load vocabulary
print("Loading vocabulary...")
with open(args.vocab, 'r') as f:
    vocab_list = [x.strip("\r\n ") for x in f]
vocab = Vocab(vocab_list)
print(f"Vocabulary loaded: {vocab.size()} entries")

# Create model
print("Creating model...")
model = JTNNVAE(
    vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG
)

# Load model weights
print(f"Loading model from {args.model}...")
try:
    weights = mx.load(args.model)
    model.load_weights(weights)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Set random seed for reproducibility
# MLX uses numpy's random state
np.random.seed(args.seed)
mx.random.seed(args.seed)

# Generate samples
print(f"\nGenerating {args.nsample} samples...")
print("=" * 60)
for i in range(args.nsample):
    try:
        smiles = model.sample_prior(prob_decode=args.prob_decode)
        print(f"{i+1:4d}: {smiles}")
    except Exception as e:
        print(f"{i+1:4d}: ERROR - {e}")
print("=" * 60)
print("Generation complete!")