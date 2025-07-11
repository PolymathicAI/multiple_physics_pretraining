import h5py
f = h5py.File("hdf5.h5", "r")
print(list(f.keys()))
for k in f.keys():
    print(k, f[k].shape)


data = f["Uvelocity"][:]
print("Shape:", data.shape)

# Test reshape
example = data[0]  # First time step
print("Example shape:", example.shape)

# Try common factors
for h in range(5, 50):
    if 400 % h == 0:
        w = 400 // h
        try:
            reshaped = example.reshape(h, w)
            print(f"Success with H={h}, W={w}")
        except:
            pass
