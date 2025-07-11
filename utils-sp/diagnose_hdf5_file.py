import os
import h5py
from collections import defaultdict

DATASET_DIR = "/lustre/scratch5/exempt/artimis/data/mpp_lsc240420_hdf5/"


def analyze_key_variability(root_dir, max_files=200):
    key_stats = defaultdict(int)
    shape_examples = {}

    h5_files = [f for f in os.listdir(root_dir) if f.endswith(".h5")]
    h5_files = h5_files[:max_files]

    print(f"Analyzing up to {len(h5_files)} files...\n")

    for fname in h5_files:
        fpath = os.path.join(root_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                keys = list(f.keys())
                for key in keys:
                    key_stats[key] += 1
                    if key not in shape_examples:
                        try:
                            shape_examples[key] = f[key].shape
                        except:
                            shape_examples[key] = "Unreadable"
        except Exception as e:
            print(f"⚠️ Error reading {fname}: {e}")

    print("🧾 Key Occurrence Summary:")
    for key, count in sorted(key_stats.items(), key=lambda x: -x[1]):
        shape = shape_examples.get(key, 'N/A')
        print(f" - {key} : seen in {count}/{len(h5_files)} files | shape: {shape}")


analyze_key_variability(DATASET_DIR)
