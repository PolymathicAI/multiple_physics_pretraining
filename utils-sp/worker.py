# worker.py
import sys
import numpy as np
import h5py
import numpy.testing as npt
import os

def convert_and_verify(npz_path, h5_path):
    with np.load(npz_path) as npz_file, h5py.File(h5_path, 'w') as h5f:
        for key in npz_file.files:
            h5f.create_dataset(key, data=npz_file[key])

    with np.load(npz_path) as npz_file, h5py.File(h5_path, 'r') as h5f:
        npz_keys = sorted(npz_file.files)
        h5_keys = sorted(h5f.keys())
        assert npz_keys == h5_keys

        for key in npz_keys:
            np_val = npz_file[key]
            h5_val = h5f[key][()] if h5f[key].shape == () else h5f[key][:]
            npt.assert_array_equal(np_val, h5_val)

    print("✅ Verified:", os.path.basename(npz_path))

if __name__ == "__main__":
    npz_file = sys.argv[1]
    h5_file = sys.argv[2]
    convert_and_verify(npz_file, h5_file)

