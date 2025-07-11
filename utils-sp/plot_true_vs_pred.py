# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:43:45 2025

@author: 368208
"""
import sys
import os

import torch
import h5py

import matplotlib             # Use a Non-Interactive Backend (Headless)
matplotlib.use('Agg')          # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np



# Add the root to sys.path to ensure that Python interpreter sees
# models/, utils/, etc., as top-level modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.avit import build_avit
from utils.YParams import YParams
from einops import rearrange



def load_checkpoint_stripped(model, checkpoint):
    model_state = model.state_dict()
    ckpt_state = checkpoint['model_state']

    # Strip projection-related layers
    keys_to_skip = ['space_bag.weight', 'debed.out_kernel', 'debed.out_bias']
    filtered_ckpt_state = {
        k: v for k, v in ckpt_state.items()
        if k in model_state and k not in keys_to_skip and model_state[k].shape == v.shape
    }

    model_state.update(filtered_ckpt_state)
    model.load_state_dict(model_state, strict=False)

    print("Loaded matching weights; skipped projection layers.")

# check the sample numbers of the test set

def check_test_set_num_samples(h5_file):
    print("Here 2 ")

    keys = sorted(h5_file.keys())  # Ensure consistent order
    print("Total samples:", len(keys))

    # Apply train/val/test split
    n = len(keys)
    train_split = int(0.8 * n)
    val_split = int(0.1 * n)
    print(f"train_split: {train_split}, val_split: {val_split}")
    test_keys = keys[train_split + val_split:]

    print("Test sample keys:", test_keys)
    return test_keys


FRAME_NUM_TO_PREDICT = 8
TEST_SAMPLE_NDX      = 5
PAST_N_STEPS         = 10

H5_PATH = os.path.expanduser('~/datasets/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5')
CKPT_PATH = "./../runs/basic_config/train_swe_diffre2d_100_epochs/training_checkpoints/best_ckpt.tar"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def main():

    h5_file = h5py.File(H5_PATH, 'r')
    test_keys = check_test_set_num_samples(h5_file)

    # ---------- Step 1: Load a sample from SWE dataset ----------
    sample_key = test_keys[TEST_SAMPLE_NDX]          # Replace with a real sample name
    time_index = FRAME_NUM_TO_PREDICT   # Assuming you want to predict the 17th frame

    traj = h5_file[sample_key]['data'][:]  # Shape: [T, H, W, C]
    print(f"Here 1 {traj}")

    # Normalize time dimension
    trajectory = torch.tensor(traj, dtype=torch.float32).permute(0, 3, 1, 2)  # [T, C, H, W]
    print(f"Here 2 {trajectory}")

    # Extract past `n_steps` input and the true target frame
    #for n_steps in(1, 2, 4, 6,8 ,10, 15, 20):
    for n_steps in range(1, 5, 1):
        for n_states in range(1, 5, 1):

            x = trajectory[time_index - n_steps:time_index]     # shape [16, C, H, W]
            y_true = trajectory[time_index]                     # shape [C, H, W]

            # Add batch dimension
            x = x.unsqueeze(1)  # [T, B, C, H, W]
            state_labels = torch.tensor([[0]])  # SWE uses single field (index 0)
            bcs = torch.tensor([[0, 0]])        # Placeholder boundary conditions

            # ---------- Step 2: Load the trained model ----------
            params = YParams('./config/mpp_avit_ti_config.yaml', 'basic_config')
            params.n_states = n_states  # only h 1 for SWE, 3 for SWE and REACDIFF

            model = build_avit(params)
            ckpt = torch.load(CKPT_PATH, map_location='cpu')
            #model.load_state_dict(ckpt['model_state'])
            load_checkpoint_stripped(model, ckpt)

            model.eval()
            print("Here 3")
            # ---------- Step 3: Run the model ----------
            with torch.no_grad():
                pred = model(x, state_labels, bcs)  # [B, C, H, W]
            print("Here 4")


            # Create the directory if it doesn't exist
            os.makedirs(RESULTS_DIR, exist_ok=True)


            # Remove batch dimension and convert to numpy
            y_pred = pred[0, 0].numpy()     # [H, W]
            y_true = y_true[0].numpy()      # [H, W]

            # ---------- Step 4: Plot ----------
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(y_true, cmap='viridis')
            axs[0].set_title('Ground Truth')
            axs[1].imshow(y_pred, cmap='viridis')
            axs[1].set_title('SWE - Model Prediction')
            plt.tight_layout()
            figname = f"true_vs_pred_{n_steps}_steps_{n_states}_states.png"
            plt.savefig(os.path.join(RESULTS_DIR, figname), dpi=300, format='png')
            #plt.show()


if __name__ == "__main__":
    main()
