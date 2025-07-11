import os
import argparse
import numpy as np
import h5py
import numpy.testing as npt
import gc
import time

def convert_npz_to_hdf5(npz_file_path, h5_file_path):
    try:
        with np.load(npz_file_path) as npz_file, h5py.File(h5_file_path, 'w') as h5f:
            for key in npz_file.files:
                h5f.create_dataset(key, data=npz_file[key])
        gc.collect()
        print(f"✅ Converted: {npz_file_path} → {h5_file_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed for {npz_file_path}: {e}")
        return False

def verify_conversion(npz_file_path, h5_file_path):
    try:
        with np.load(npz_file_path) as npz_data, h5py.File(h5_file_path, 'r') as h5f:
            npz_keys = sorted(npz_data.files)
            h5_keys = sorted(list(h5f.keys()))

            if npz_keys != h5_keys:
                print("❌ Dataset keys do not match!")
                print("NPZ keys:", npz_keys)
                print("H5 keys:", h5_keys)
                return False
            else:
                print("✅ Dataset keys match.")

            for key in npz_keys:
                try:
                    np_val = npz_data[key]
                    h5_val = h5f[key][()] if h5f[key].shape == () else h5f[key][:]
                    npt.assert_array_equal(np_val, h5_val)
                    print(f"✅ Data for key '{key}' matches.")
                except AssertionError as e:
                    print(f"❌ Data mismatch for key '{key}': {e}")
                    return False
                except Exception as e:
                    print(f"⚠️ Error comparing key '{key}': {e}")
                    return False
        gc.collect()
        print("🎉 All data verified successfully!")
        return True
    except Exception as e:
        print(f"❌ Verification failed for {npz_file_path}: {e}")
        return False

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    if not npz_files:
        print("🚫 No .npz files found in input directory.")
        return

    for idx, filename in enumerate(npz_files, 1):
        print(f"\n📁 [{idx}/{len(npz_files)}] Processing '{filename}'...")
        npz_path = os.path.join(input_dir, filename)
        basename, _ = os.path.splitext(filename)
        h5_path = os.path.join(output_dir, basename + '.h5')

        success = convert_npz_to_hdf5(npz_path, h5_path)
        if success:
            verified = verify_conversion(npz_path, h5_path)
            if not verified:
                print(f"❗ Verification failed for {filename}.")
        else:
            print(f"⛔ Skipping verification for {filename} due to conversion failure.")

        time.sleep(0.1)  # brief delay to manage IO pressure

def main():
    parser = argparse.ArgumentParser(description="Convert and verify .npz files to .h5 format")
    parser.add_argument("--input_dir", required=True, help="Directory containing .npz files")
    parser.add_argument("--output_dir", required=True, help="Directory to save .h5 files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

