"""
Download and save MOSES dataset for JTNN training.
MOSES (Molecular Sets) is a benchmark dataset for molecular generation.
The original dataset has 2 million molecules, but we will use a subset of 100,000 molecules.
"""

from datasets import load_dataset
import os
import random

# Configuration: sample 100K molecules total
TRAIN_SIZE = 90_000
TEST_SIZE = 10_000
TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE

# Set random seed for reproducibility
random.seed(67)

# Load MOSES dataset
print("Loading MOSES dataset from HuggingFace...")
train_full = load_dataset("katielink/moses", split="train")
test_full = load_dataset("katielink/moses", split="test")

# Print original dataset information
print("\nOriginal Dataset Information:")
print(f"  Train size: {len(train_full):,} molecules")
print(f"  Test size: {len(test_full):,} molecules")
print(f"  Total: {len(train_full) + len(test_full):,} molecules")
print(f"  Features: {list(train_full.features.keys())}")
print(f"  Sample SMILES: {train_full[0]['SMILES']}")

# Sample subsets
print(f"\nSampling {TOTAL_SIZE:,} molecules ({TRAIN_SIZE:,} train, {TEST_SIZE:,} test)...")
train_indices = random.sample(range(len(train_full)), min(TRAIN_SIZE, len(train_full)))
test_indices = random.sample(range(len(test_full)), min(TEST_SIZE, len(test_full)))

train = train_full.select(train_indices)
test = test_full.select(test_indices)

print(f"  Sampled {len(train):,} train molecules")
print(f"  Sampled {len(test):,} test molecules")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save train set
train_file = "data/moses_train.txt"
print(f"\nSaving train set to {train_file}...")
with open(train_file, 'w') as f:
    for example in train:
        f.write(example['SMILES'] + '\n')
print(f"  Saved {len(train):,} SMILES strings")

# Save test set
test_file = "data/moses_test.txt"
print(f"\nSaving test set to {test_file}...")
with open(test_file, 'w') as f:
    for example in test:
        f.write(example['SMILES'] + '\n')
print(f"  Saved {len(test):,} SMILES strings")

print("\n✅ Dataset download and save complete!")
print(f"\nNext steps:")
print(f"  1. Preprocess: python Scripts/preprocess.py -t {train_file} -n 10 -j 4")
print(f"  2. Create vocab from the preprocessed data")
print(f"  3. Train: python Scripts/train.py --train . --vocab vocab.txt --save_dir models")