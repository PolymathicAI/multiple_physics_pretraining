"""
Remember to parameterize the file paths eventually
"""
import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import os
try:
    from .mixed_dset_sampler import MultisetSampler
    from .hdf5_datasets import *
except ImportError:
    from .mixed_dset_sampler import MultisetSampler
    from .hdf5_datasets import *
import os
import glob

broken_paths = []
# IF YOU ADD A NEW DSET MAKE SURE TO UPDATE THIS MAPPING SO MIXED DSET KNOWS HOW TO USE IT
DSET_NAME_TO_OBJECT = {
            'lsc240420': LSC240420Dataset,
            'swe': SWEDataset,
            'incompNS': IncompNSDataset,
            'diffre2d': DiffRe2DDataset,
            'compNS': CompNSDataset,
            }

def get_data_loader(params, paths, distributed, split='train', rank=0, train_offset=0):
    # paths, types, include_string = zip(*paths)
    dataset = MixedDataset(paths, n_steps=params.n_steps, train_val_test=params.train_val_test, split=split,
                            tie_fields=params.tie_fields, use_all_fields=params.use_all_fields, enforce_max_steps=params.enforce_max_steps,
                            train_offset=train_offset)
    # dataset = IncompNSDataset(paths[0], n_steps=params.n_steps, train_val_test=params.train_val_test, split=split)
    seed = torch.random.seed() if 'train'==split else 0
    if distributed:
        base_sampler = DistributedSampler
    else:
        base_sampler = RandomSampler
    sampler = MultisetSampler(dataset, base_sampler, params.batch_size,
                               distributed=distributed, max_samples=params.epoch_size,
                               rank=rank)
    # sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False, #(sampler is None),
                            sampler=sampler, # Since validation is on a subset, use a fixed random subset,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())
    return dataloader, dataset, sampler


class MixedDataset(Dataset):
    def __init__(self, path_list=[], n_steps=1, dt=1, train_val_test=(.8, .1, .1),
                  split='train', tie_fields=True, use_all_fields=True, extended_names=False,
                  enforce_max_steps=False, train_offset=0):
        super().__init__()
        # Global dicts used by Mixed DSET.
        self.train_offset = train_offset
        self.path_list, self.type_list, self.include_string = zip(*path_list)
        self.tie_fields = tie_fields
        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test
        self.use_all_fields = use_all_fields

        for dset, path, include_string in zip(self.type_list, self.path_list, self.include_string):
            subdset = DSET_NAME_TO_OBJECT[dset](path, include_string, n_steps=n_steps,
                                                 dt=dt, train_val_test=train_val_test, split=split)
            # Check to make sure our dataset actually exists with these settings
            try:
                len(subdset)
            except ValueError:
                raise ValueError(f'Dataset {path} is empty. Check that n_steps < trajectory_length in file.')
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1]+len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()

        print("\n[DEBUG] Final subset_dict:")
        for k, v in self.subset_dict.items():
            print(f"  {k}: {v}")

    def get_state_names(self):
        name_list = []
        if self.use_all_fields:
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                name_list += field_names
            return name_list
        else:
            visited = set()
            for dset in self.sub_dsets:
                    name = dset.get_name() # Could use extended names here
                    if not name in visited:
                        visited.add(name)
                        name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl] # Flatten the names

    def _build_subset_dict(self):
        # Maps fields to subsets of variables
        if self.tie_fields: # Hardcoded, but seems less effective anyway
            subset_dict = {
                        'swe': [3],
                        'incompNS': [0, 1, 2],
                        'compNS': [0, 1, 2, 3],
                        'diffre2d': [4, 5]
                        }
        elif self.use_all_fields:
            cur_max = 0
            subset_dict = {}
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
        else:
            subset_dict = {}
            cur_max = self.train_offset
            for dset in self.sub_dsets:
                name = dset.get_name(self.extended_names)
                if not name in subset_dict:
                    subset_dict[name] = list(range(cur_max, cur_max + len(dset.field_names)))
                    cur_max += len(dset.field_names)
        return subset_dict

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.offsets, index, side='right')-1 #which dataset are we are on
        local_idx = index - max(self.offsets[file_idx], 0)
        try:
            x, bcs, y = self.sub_dsets[file_idx][local_idx]
        except:
            print('FAILED AT ', file_idx, local_idx, index,int(os.environ.get("RANK", 0)))
            thisvariabledoesntexist
        return x, file_idx, torch.tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]), bcs, y

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
