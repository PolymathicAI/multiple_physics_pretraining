import sys
import numpy as np
import h5py
import numpy.testing as npt
import os
import gc

def convert_npz_to_h5(npz_file, h5_file):
    npz_data = None
    h5f = None

    try:
        with np.load(npz_file) as npz_data, h5py.File(h5_file, 'w') as h5f:
            for key in npz_data.files:
                h5f.create_dataset(key, data=npz_data[key])
        print(f"\t✅ Converted {npz_file} → {h5_file}")
    finally:
        del npz_data, h5f
        gc.collect()

def verify_conversion(npz_file, h5_file):
    npz_data = None
    h5f = None

    try:
        with np.load(npz_file) as npz_data, h5py.File(h5_file, 'r') as h5f:
            npz_keys = sorted(npz_data.files)
            h5_keys = sorted(h5f.keys())

            if npz_keys != h5_keys:
                print("❌ Verification failed: Key mismatch")
                print("NPZ keys:", npz_keys)
                print("H5 keys:", h5_keys)
                return False

            for key in npz_keys:
                np_val = npz_data[key]
                h5_val = h5f[key][()] if h5f[key].shape == () else h5f[key][:]
                try:
                    npt.assert_array_equal(np_val, h5_val)
                except AssertionError as e:
                    print(f"❌ Verification failed for key '{key}': {e}")
                    return False
                finally:
                    del np_val, h5_val
        print(f"\t✅ Verified: {npz_file}")
        return True
    finally:
        del npz_data, h5f, npz_keys, h5_keys, key
        gc.collect()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_one.py input.npz output.h5")
        sys.exit(1)

    npz_file = sys.argv[1]
    h5_file = sys.argv[2]

    try:
        convert_npz_to_h5(npz_file, h5_file)
        verified = verify_conversion(npz_file, h5_file)
        if not verified:
            sys.exit(2)  # Verification failed
    except Exception as e:
        print(f"❌ Exception during processing: {e}")
        sys.exit(3)

