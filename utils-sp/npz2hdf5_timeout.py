import os
import subprocess
import argparse

def run_worker_with_timeout(npz_path, h5_path, timeout=10):
    try:
        result = subprocess.run(
            ["python3", "worker.py", npz_path, h5_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"⚠️ Error in worker:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout: Skipping {os.path.basename(npz_path)} after {timeout} seconds.")
    except Exception as e:
        print(f"❌ Subprocess error: {e}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    skip_file = "lsc240420_id04427_pvi_idx00078.npz"
    npz_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npz"))

    for idx, filename in enumerate(npz_files, 1):
        if filename == skip_file:
            print(f"⏭️ Skipping manually excluded file: {filename}")
            continue

        print(f"\n🔄 [{idx}/{len(npz_files)}] Processing: {filename}")
        npz_path = os.path.join(input_dir, filename)
        h5_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".h5")
        run_worker_with_timeout(npz_path, h5_path, timeout=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

