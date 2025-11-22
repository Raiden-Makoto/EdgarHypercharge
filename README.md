# EdgarHypercharge - JTNN VAE for Molecular Generation

A Junction Tree Variational Autoencoder (JTNN VAE) implementation for molecular generation, converted from PyTorch to Apple MLX for efficient training and inference on Apple Silicon.

## Overview

This project implements a Junction Tree Neural Network Variational Autoencoder (JTNN VAE) for generating molecular structures. The model decomposes molecules into junction trees (cliques) and learns to generate valid molecular structures through a tree-based encoding-decoding process.

**Key Features:**
- ✅ **Python 3** compatible (converted from Python 2)
- ✅ **Apple MLX** framework for native Apple Silicon acceleration
- ✅ **Optimized for efficiency**: Default configuration uses ~81K parameters
- ✅ **Complete pipeline**: Preprocessing, training, and inference scripts
- ✅ **Memory efficient**: ~0.3 MB model size

## Installation

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3) for MLX acceleration
- macOS 12.0+ (for MLX)

## Acknowledgments

- Original JTNN implementation: [wengong-jin/icml18-jtnn](https://github.com/wengong-jin/icml18-jtnn)
- Apple MLX framework: [ml-explore/mlx](https://github.com/ml-explore/mlx)

