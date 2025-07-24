# File: data_utils/npz_datasets.py

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

class LscNpzDataset(Dataset):
    def __init__(self, path, include_string='', n_steps=5, dt=1, split='train',
                 train_val_test=(0.8, 0.1, 0.1), subname=None, extra_specific=False):

        self.root_dir = path
        self.n_steps = n_steps
        self.dt = dt
        self.split = split
        self.train_val_test = train_val_test
        self.include_string = include_string
        self.type = 'lsc_npz'
        self.field_names = self._specifics()[2]
        self.title = 'lsc_npz'

        self.file_list = sorted([
            f for f in os.listdir(self.root_dir)
            if f.endswith(".npz") and include_string in f
        ], key=self._extract_timestep)

        # Determine split range
        total = len(self.file_list) - (n_steps + dt - 1)
        train_end = int(train_val_test[0] * total)
        val_end = train_end + int(train_val_test[1] * total)

        if split == 'train':
            self.indices = range(0, train_end)
        elif split == 'val':
            self.indices = range(train_end, val_end)
        else:
            self.indices = range(val_end, total)

    def _extract_timestep(self, filename):
        match = re.search(r"idx(\d+)", filename)
        return int(match.group(1)) if match else -1

    def __len__(self):
        return len(self.indices)

    def get_name(self, full_name=False):
        return "lsc_npz"  # or whatever label you want to appear in logs

    def __getitem__(self, idx):
        idx = self.indices[idx]
        input_files = self.file_list[idx : idx + self.n_steps]
        target_file = self.file_list[idx + self.n_steps]

        # Fields to use as input/output channels
        selected_fields = [
                           'pressure_throw', 'density_throw', 'temperature_throw',
                           'density_case', 'pressure_case', 'temperature_case',
                           'Uvelocity', 'Wvelocity'
                          ]

        def load_tensor(fpath):
            with np.load(fpath) as data:
                arrays = []
                for key in selected_fields:
                    if key in data:
                        arr = data[key]
                        try:
                            arr = arr.astype(np.float32)
                        except Exception as e:
                            raise ValueError(f"Could not convert {key} in {fpath} to float32: {e}")

                        if arr.ndim == 2:
                            arrays.append(arr)
                        else:
                            raise ValueError(f"{key} in {fpath} is not 2D")
                    else:
                        raise KeyError(f"{key} missing in {fpath}")
                #stacked = np.stack(arrays, axis=0)  # shape: [C, H, W]
                stacked = np.stack(arrays, axis=0)[:, :560, :192]

            return torch.tensor(stacked, dtype=torch.float32)

        # Load input sequence
        input_tensors = [load_tensor(os.path.join(self.root_dir, f)) for f in input_files]
        x = torch.stack(input_tensors, dim=0)  # shape: [T, C, H, W]

        # Load target
        y = load_tensor(os.path.join(self.root_dir, target_file))  # shape: [C, H, W]
        bcs = torch.zeros(2)  # placeholder

        return x, bcs, y

    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = [
            'pressure_throw', 'density_throw', 'temperature_throw',
            'density_case', 'pressure_case', 'temperature_case',
            'Uvelocity', 'Wvelocity'
        ]
        type = 'lsc_npz'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f=None):
        total_timesteps = len(self.file_list)
        return 1, total_timesteps

    def _get_specific_bcs(self, f=None):
        return [0, 0]  # 0 = non-periodic, 1 = periodic

    def get_per_file_dsets(self):
        # No sub-files — treat whole dataset as one logical file
        return [self]

