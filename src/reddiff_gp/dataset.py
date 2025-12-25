import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GPDataset(Dataset):
	"""
	Gaussian Process Dataset for RED-diff

	Args:
	root: Root directory containing the dataset
	name: Dataset name (used to construct file paths)
	idx: Index, slice, or list for selecting specific samples
	transform: Optional transform to apply to inputs
	target_transform: Optional transform to apply to targets
	"""
	def __init__(self, root, name, idx):
		self.root = root
		self.name = name
		self.index = idx

		# Convert idx to list of indices
		if isinstance(idx, int):
			self.original_indices = [idx]
		elif isinstance(idx, slice):
			# Load full array to know size
			dataset_dir = root  # Files are directly in root directory
			full_array = np.load(os.path.join(dataset_dir, f"test_inputs_{name}.npy"))
			self.original_indices = list(range(len(full_array)))[idx]
		elif isinstance(idx, list):
			self.original_indices = idx
		else:
			raise TypeError(f"idx must be int, slice, or list, got {type(idx)}")

		# dataset_dir = os.path.join(root, f"./results/datasets_{name}")
		dataset_dir = root
		# Use regular indexing (works with list of indices)
		self.true_inputs = np.load(os.path.join(dataset_dir, f"test_inputs_{name}.npy"))[self.original_indices]
		self.obv_targets = np.load(os.path.join(dataset_dir, f"test_targets_{name}.npy"))[self.original_indices]
		# Ensure 2D (in case single index)
		if self.true_inputs.ndim == 1:
			self.true_inputs = self.true_inputs.reshape(1, -1)
			self.obv_targets = self.obv_targets.reshape(1, -1)
		self.true_inputs = torch.from_numpy(self.true_inputs).float()
		self.obv_targets = torch.from_numpy(self.obv_targets).float()

	def __len__(self):
		return len(self.true_inputs)

	def __getitem__(self, idx):
		return self.true_inputs[idx], self.obv_targets[idx], {'idx': self.original_indices[idx]}
	

def get_gp_dataset_loader(root, name, idx, **kwargs):
	# dataset_dir = os.path.join(root , f"./results/datasets_{name}")
	# true_inputs = np.load(os.path.join(dataset_dir, f"test_inputs_{name}.npy"))[idx]
	# obv_targets = np.load(os.path.join(dataset_dir, f"test_targets_{name}.npy"))[idx]
	
	dataset = GPDataset(root, name, idx)
	
	batch_size = kwargs.get('batch_size', 1)
	shuffle = kwargs.get('shuffle', False)

	loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle
	)
	return loader


def build_loader(cfg, dataset_attr='dataset'):
    assert type(dataset_attr) != list
    dset_cfg = getattr(cfg, dataset_attr)
    root, name, idx, list_idx = dset_cfg.root, dset_cfg.name, dset_cfg.index, dset_cfg.list

	        

    # Convert integer index to range 0..index (inclusive)
    if isinstance(idx, int):
        idx = list(range(idx + 1))
        print(f"Converted integer index to list range: {idx}")

    if not list_idx:
        idx = list(range(dset_cfg.index, dset_cfg.index + 1))
    loader = get_gp_dataset_loader(idx=idx, **dset_cfg)

    return loader, dset_cfg.columns
