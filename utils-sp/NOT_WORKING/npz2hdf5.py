import numpy as np
import h5py
import numpy.testing as npt
import os

def convert_npz_to_hdf5(filename):
    # Load the .npz file
    basename, _ = os.path.splitext(filename)
    npz_file = np.load(filename)

    # Create an HDF5 file
    with h5py.File(f"{basename}.h5", 'w') as h5f:
        for key in npz_file.files:
            h5f.create_dataset(key, data=npz_file[key])
    print(f"✅ Converted '{filename}' → '{basename}.h5'")

def verify_conversion(npz_file_path, h5_file_path):
    npz_data = np.load(npz_file_path)
    with h5py.File(h5_file_path, 'r') as h5f:
        npz_keys = sorted(npz_data.files)
        h5_keys = sorted(list(h5f.keys()))

        # Check if keys match
        if npz_keys != h5_keys:
            print("❌ Dataset keys do not match!")
            print("NPZ keys:", npz_keys)
            print("H5 keys:", h5_keys)
            return False
        else:
            print("✅ Dataset keys match.")

        # Check if data matches
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

    print("🎉 All data verified successfully!")
    return True

def main():
    filename = "lsc240420_id01767_pvi_idx00086.npz"
    basename, _ = os.path.splitext(filename)

    convert_npz_to_hdf5(filename)
    verify_conversion(filename, f"{basename}.h5")

if __name__ == "__main__":
    main()

