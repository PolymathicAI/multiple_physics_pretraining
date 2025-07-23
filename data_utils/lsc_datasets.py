# File: data_utils/npz_datasets.py

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

class LscNpzDataset(Dataset):
    def __init__(self, root_dir, n_steps=5, dt=1, split='train',
                 train_val_test=(0.8, 0.1, 0.1), include_string=''):
        self.root_dir = root_dir
        self.n_steps = n_steps
        self.dt = dt
        self.split = split
        self.train_val_test = train_val_test
        self.include_string = include_string

        self.file_list = sorted([
            f for f in os.listdir(root_dir)
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

