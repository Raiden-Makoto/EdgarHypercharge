# Model Parameter Analysis

## Default Parameters

Current default values in `Scripts/train.py` and `Scripts/inference.py`:

- **hidden_size**: 32 (reduced from 450)
- **latent_size**: 8 (reduced from 56, becomes 4 after division by 2 for tree/mol)
- **depthT**: 3 (reduced from 20, tree encoder depth)
- **depthG**: 2 (reduced from 3, graph encoder depth)
- **batch_size**: 32

**Note**: These defaults are optimized for **≤100K parameters** to enable efficient training and inference on resource-constrained systems.

## Parameter Count Estimate

For a vocabulary size of ~500 SMILES (typical for smaller datasets):

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Embeddings (2x) | ~32K | 39.5% |
| JTNNDecoder | ~27K | 33.6% |
| JTMPN (2x) | ~13K | 15.9% |
| JTNN Encoder | ~8K | 10.3% |
| VAE layers | ~656 | 0.8% |
| **TOTAL** | **~81K** | **100%** |

**For larger vocabularies:**
- vocab_size=1000: ~113K parameters (slightly over target)
- vocab_size=2000: ~195K parameters

## Memory Usage

- **Model weights** (float32): ~0.32 MB
- **Training** (with gradients): ~0.64 MB
- **Inference**: ~0.32 MB

## Assessment

✅ **Defaults are optimized for minimal resource usage**:
- ~81K parameters (99.4% reduction from original 13.4M)
- Extremely low memory footprint
- Suitable for edge devices, mobile, and resource-constrained systems
- May have reduced model capacity compared to larger models

## Trade-offs with Reduced Parameters

The current defaults prioritize efficiency over capacity:

**Advantages:**
- ✅ Extremely low memory usage (~0.3 MB)
- ✅ Fast training and inference
- ✅ Suitable for edge devices
- ✅ Lower computational requirements

**Potential Limitations:**
- ⚠️ Reduced model capacity may affect generation quality
- ⚠️ May require more training epochs to converge
- ⚠️ Smaller vocabulary may limit molecular diversity

## Increasing Parameters (if needed)

If you need more model capacity, you can increase:

1. **Increase hidden_size**: 32 → 64 or 128
   - 64: ~200K parameters
   - 128: ~600K parameters

2. **Increase latent_size**: 8 → 16 or 32
   - Provides richer latent representations

3. **Increase depthT**: 3 → 5 or 10
   - Better long-range dependencies
   - Minimal parameter increase (~5-10%)

4. **Increase vocab_size**: Depends on dataset
   - Larger vocab = more embedding parameters
   - Must match your training data

## Recommendations

The current defaults are **optimal** for:
- Resource-constrained systems (<1GB RAM)
- Edge devices and mobile applications
- Fast prototyping and experimentation
- Small to medium datasets (<10K molecules)

Consider increasing defaults if:
- You have sufficient computational resources
- Training on large datasets (>100K molecules)
- Need higher generation quality
- Running on systems with 8GB+ RAM

