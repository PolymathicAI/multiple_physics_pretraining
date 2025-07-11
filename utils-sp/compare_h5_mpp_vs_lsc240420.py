import h5py

import h5py

def inspect_h5_file(path):
    with h5py.File(path, 'r') as f:
        print(f"\n🔍 File: {path}")
        print("Top-level keys:", list(f.keys()))

        for key in f.keys():
            obj = f[key]
            print(f"\n➤ Key: {key} ({type(obj)})")
            if isinstance(obj, h5py.Group):
                print("   └─ Group with nested keys:", list(obj.keys()))
                if 'data' in obj:
                    dset = obj['data']
                    print(f"   └─ 'data' shape: {dset.shape}")
                    print(f"   └─ dtype: {dset.dtype}")
            elif isinstance(obj, h5py.Dataset):
                print(f"   └─ Dataset shape: {obj.shape}")
                print(f"   └─ dtype: {obj.dtype}")
            else:
                print("   └─ Unknown HDF5 object type")


inspect_h5_file("swe.h5")
inspect_h5_file("diffrec.h5")
inspect_h5_file("lsc.h5")
