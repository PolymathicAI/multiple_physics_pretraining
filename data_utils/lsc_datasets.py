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

    def __getitem__(self, idx):
        idx = self.indices[idx]
        input_files = self.file_list[idx : idx + self.n_steps]
        target_file = self.file_list[idx + self.n_steps]

        inputs = []
        for f in input_files:
            data = np.load(os.path.join(self.root_dir, f))["arr_0"]
            inputs.append(torch.tensor(data, dtype=torch.float32))

        target = np.load(os.path.join(self.root_dir, target_file))["arr_0"]
        target = torch.tensor(target, dtype=torch.float32)

        x = torch.stack(inputs, dim=0)     # shape: [T, C, H, W]
        y = target                         # shape: [C, H, W]

        # No BCs used in your case
        bcs = torch.zeros(2)  # placeholder for compatibility

        return x, bcs, y

    @staticmethod
    def _specifics():
        time_index = 0                     # usually 0 if your data is [T, C, H, W]
        sample_index = None               # if not multiple samples per file
        field_names = ['v1', 'v2', ..., 'v8']  # name of your 8 channels
        type = 'lsc_npz'
        split_level = 'sample'            # or 'file' if you split by file
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f=None):
        total_timesteps = len(self.file_list)
        return 1, total_timesteps

    def _get_specific_bcs(self, f=None):
        return [0, 0]  # 0 = non-periodic, 1 = periodic

