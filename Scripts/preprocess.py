"""
Preprocessing script for JTNN.
Converts SMILES strings to molecular tree structures.
"""

from multiprocessing import Pool
import argparse
import os
from JTNN.moltree import MolTree
import pickle
import rdkit


def tensorize(smiles, assm=True):
    """
    Convert SMILES string to molecular tree structure.
    
    Args:
        smiles: SMILES string
        assm: Whether to assemble candidates
    
    Returns:
        MolTree object with molecules removed (for serialization), or None if error
    """
    try:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)

        # Remove RDKit molecule objects for serialization
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol

        return mol_tree
    except Exception:
        # Return None for invalid SMILES - will be filtered out
        return None


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser(
        description='Preprocess SMILES data for JTNN training'
    )
    parser.add_argument(
        "-t", "--train", dest="train_path", required=True,
        help="Path to training data file (one SMILES per line)"
    )
    parser.add_argument(
        "-n", "--split", dest="nsplits", type=int, default=10,
        help="Number of splits to create (default: 10)"
    )
    parser.add_argument(
        "-j", "--jobs", dest="njobs", type=int, default=2,
        help="Number of parallel jobs (default: 2)"
    )
    args = parser.parse_args()

    pool = Pool(args.njobs)
    num_splits = args.nsplits

    with open(args.train_path, 'r') as f:
        data = [
            line.strip("\r\n ").split()[0]
            for line in f
            if line.strip() and line.strip().split()
        ]

    print(f"Processing {len(data)} SMILES strings...")
    results = pool.map(tensorize, data)
    # Filter out None results (invalid SMILES)
    all_data = [r for r in results if r is not None]
    skipped = len(results) - len(all_data)
    if skipped > 0:
        print(f"Skipped {skipped} invalid SMILES strings")
    print(f"Processed {len(all_data)} molecular trees")

    # Create output directory
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving preprocessed data to {output_dir}/")

    # Calculate split size (ceiling division)
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st:st + le]

        output_file = os.path.join(output_dir, f'tensors-{split_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(sub_data)} trees to {output_file}")

    pool.close()
    pool.join()
    print("Preprocessing complete!")
