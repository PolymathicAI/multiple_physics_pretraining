import os
import argparse
import numpy as np
import h5py
import numpy.testing as npt
import gc
import time

def convert_and_verify_single_file(npz_path, h5_path):
    npz_data = None
    h5f = None

    try:
        # === Conversion ===
        with np.load(npz_path) as npz_file, h5py.File(h5_path, 'w') as h5f:
            for key in npz_file.files:
                h5f.create_dataset(key, data=npz_file[key])
        print(f"✅ Converted: {npz_path} → {h5_path}")

        # === Verification ===
        with np.load(npz_path) as npz_data, h5py.File(h5_path, 'r') as h5f:
            npz_keys = sorted(npz_data.files)
            h5_keys = sorted(h5f.keys())

            if npz_keys != h5_keys:
                print("❌ Keys mismatch")
                print("NPZ keys:", npz_keys)
                print("H5 keys:", h5_keys)
                return

            print("✅ Keys match")
            for key in npz_keys:
                np_val = npz_data[key]
                h5_val = h5f[key][()] if h5f[key].shape == () else h5f[key][:]
                npt.assert_array_equal(np_val, h5_val)
                print(f"✅ Data match: '{key}'")

        print("🎉 File verified successfully")

    except Exception as e:
        print(f"❌ Error processing {npz_path}: {e}")

    finally:
        # Force cleanup of variables and file handles
        del npz_data, h5f, np_val, h5_val, key
        gc.collect()
        time.sleep(0.1)  # small pause to ease I/O stress

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    if not npz_files:
        print("🚫 No .npz files found.")
        return

    for idx, filename in enumerate(npz_files, 1):
        if filename == "lsc240420_id04427_pvi_idx00078.npz":
            print("\nSkipping lsc240420_id04427_pvi_idx00078.npz")
            time.sleep(1.0)  # small pause to ease I/O stress
            continue

        print(f"\n🔄 [{idx}/{len(npz_files)}] Processing: {filename}")
        npz_path = os.path.join(input_dir, filename)
        h5_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.h5')

        convert_and_verify_single_file(npz_path, h5_path)

def main():
    parser = argparse.ArgumentParser(description="Convert .npz to .h5 with resource isolation")
    parser.add_argument("--input_dir", required=True, help="Directory with .npz files")
    parser.add_argument("--output_dir", required=True, help="Directory to write .h5 files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

