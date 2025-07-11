import os
import argparse
import numpy as np
import h5py
import numpy.testing as npt
import gc
import time
import multiprocessing

# === Convert & Verify ===
def convert_and_verify_single_file(npz_path, h5_path):
    npz_data = None
    h5f = None

    try:
        with np.load(npz_path) as npz_file, h5py.File(h5_path, 'w') as h5f:
            for key in npz_file.files:
                h5f.create_dataset(key, data=npz_file[key])
        print(f"✅ Converted: {npz_path} → {h5_path}")

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
        del npz_data, h5f
        gc.collect()
        time.sleep(0.1)

# === Top-Level Function for multiprocessing ===
def conversion_task(queue, npz_path, h5_path):
    try:
        convert_and_verify_single_file(npz_path, h5_path)
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {e}")

# === Timeout Wrapper with Safe Cleanup ===
def run_with_timeout(npz_path, h5_path, timeout=10):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=conversion_task, args=(queue, npz_path, h5_path))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print(f"⏱️ Timeout: Skipping {os.path.basename(npz_path)} after {timeout} seconds.")
        p.terminate()
        p.join()
        queue.close()
        queue.cancel_join_thread()
        return  # Don't try to get from queue!

    try:
        result = queue.get_nowait()
        if result != "done":
            print(f"⚠️ Error: {result}")
    except Exception:
        print(f"⚠️ No result received for {os.path.basename(npz_path)} — assuming failure.")
    finally:
        queue.close()
        queue.cancel_join_thread()

# === Directory Processor ===
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    if not npz_files:
        print("🚫 No .npz files found.")
        return

    skip_file = "lsc240420_id04427_pvi_idx00078.npz"

    for idx, filename in enumerate(npz_files, 1):
        if filename == skip_file:
            print(f"⏭️ Skipping manually excluded file: {filename}")
            continue

        print(f"\n🔄 [{idx}/{len(npz_files)}] Processing: {filename}")
        npz_path = os.path.join(input_dir, filename)
        h5_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.h5')

        run_with_timeout(npz_path, h5_path, timeout=10)

# === CLI Entry ===
def main():
    parser = argparse.ArgumentParser(description="Convert .npz to .h5 with timeout and verification")
    parser.add_argument("--input_dir", required=True, help="Directory with .npz files")
    parser.add_argument("--output_dir", required=True, help="Directory to write .h5 files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

