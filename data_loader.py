import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable


class PhiSatDataset(Dataset):
    """
    PyTorch Dataset for loading data from a Zarr dataset created with add_sample function.
    
    Args:
        zarr_path (str): Path to the Zarr dataset.
        dataset_set (str): One of {"trainval", "test"} (case-insensitive).
        transform (callable, optional): Transform to apply to the images.
        target_transform (callable, optional): Transform to apply to the labels.
        task_filter (str, optional): If provided, only load samples with this task.
        metadata_keys (list, optional): List of metadata keys to include in sample.
    """
    def __init__(
        self,
        zarr_path: str,
        dataset_set: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        task_filter: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.root = zarr.open(zarr_path, mode='r')
        self.dataset_set = dataset_set.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.task_filter = task_filter.lower() if task_filter else None
        self.metadata_keys = metadata_keys or []
        self.verbose = verbose
        
        # Verify the dataset exists
        if self.dataset_set not in self.root:
            raise ValueError(f"Dataset set '{dataset_set}' not found in Zarr store")
            
        self.dataset_group = self.root[self.dataset_set]
        
        # Get all sample IDs
        self.sample_ids = sorted(self.dataset_group.keys())
        
        if self.verbose:
            print(f"Found {len(self.sample_ids)} total samples in {dataset_set}")
            
        # Get available tasks for debugging
        if self.verbose:
            tasks = {}
            for sid in self.sample_ids[:min(100, len(self.sample_ids))]:
                task = self.dataset_group[sid].attrs.get('task', '')
                tasks[task] = tasks.get(task, 0) + 1
            print(f"Available tasks in first 100 samples: {tasks}")
        
        # Filter by task if needed
        if self.task_filter:
            original_count = len(self.sample_ids)
            self.sample_ids = [
                sid for sid in self.sample_ids 
                if self.dataset_group[sid].attrs.get('task', '') == self.task_filter
            ]
            if self.verbose:
                print(f"Filtered to {len(self.sample_ids)} samples with task '{self.task_filter}'")
                
            if len(self.sample_ids) == 0:
                available_tasks = set()
                for sid in list(self.dataset_group.keys())[:min(100, len(list(self.dataset_group.keys())))]:
                    available_tasks.add(self.dataset_group[sid].attrs.get('task', ''))
                raise ValueError(
                    f"No samples found with task '{self.task_filter}'. "
                    f"Available tasks: {available_tasks}"
                )
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]
        sample_group = self.dataset_group[sample_id]
        
        # Load image and label
        img = sample_group['img'][:]  # This loads as numpy array
        label = sample_group['label'][:]
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
            
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(label)
        
        # Get task
        task = sample_group.attrs.get('task', '')
        
        # Create sample dict
        sample = {
            'img': img,
            'label': label,
            'task': task,
            'sample_id': sample_id,
        }
        
        # Add requested metadata
        for key in self.metadata_keys:
            if key in sample_group.attrs:
                sample[key] = sample_group.attrs[key]
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle different task types and metadata.
    
    Args:
        batch: List of sample dictionaries from PhiSatDataset
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Group samples by task to handle different label shapes
    task_groups = {}
    for sample in batch:
        task = sample['task']
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(sample)
    
    result = {}
    
    # Process each task group separately
    for task, samples in task_groups.items():
        # Get all keys from the first sample
        keys = samples[0].keys()
        
        for key in keys:
            # Skip task and sample_id for batching
            if key in ['task', 'sample_id']:
                continue
                
            # Handle tensors
            if isinstance(samples[0][key], torch.Tensor):
                # Stack tensors with same shapes
                try:
                    result[f"{task}_{key}"] = torch.stack([s[key] for s in samples])
                except RuntimeError:
                    # If tensors have different shapes, return as list
                    result[f"{task}_{key}"] = [s[key] for s in samples]
            else:
                # For non-tensor data, collect as list
                result[f"{task}_{key}"] = [s[key] for s in samples]
        
        # Store sample IDs
        result[f"{task}_sample_ids"] = [s['sample_id'] for s in samples]
        
    # Store task information
    result['tasks'] = list(task_groups.keys())
    result['task_counts'] = {task: len(samples) for task, samples in task_groups.items()}
    
    return result


def get_zarr_dataloader(
    zarr_path: str,
    dataset_set: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    task_filter: Optional[str] = None,
    metadata_keys: Optional[List[str]] = None,
    verbose: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for Zarr dataset.
    
    Args:
        zarr_path: Path to Zarr dataset
        dataset_set: One of {"trainval", "test"}
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of data loading workers
        transform: Transform to apply to images
        target_transform: Transform to apply to labels
        task_filter: Filter to specific task (classification, segmentation, etc.)
        metadata_keys: Keys of metadata to include
        verbose: Whether to print debug information
        
    Returns:
        PyTorch DataLoader for the dataset
    """
    dataset = PhiSatDataset(
        zarr_path=zarr_path,
        dataset_set=dataset_set,
        transform=transform,
        target_transform=target_transform,
        task_filter=task_filter,
        metadata_keys=metadata_keys,
        verbose=verbose
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

# Example normalization transform
class NormalizeChannels:
    """Normalize each channel to [0, 1] or using mean/std"""
    def __init__(self, means=None, stds=None, min_max=True):
        self.means = means
        self.stds = stds
        self.min_max = min_max
        
    def __call__(self, img):
        """
        Args:
            img: numpy array of shape (C, H, W)
        """
        if self.min_max:
            # Min-max normalization per channel
            result = torch.from_numpy(img.copy())
            for c in range(img.shape[0]):
                min_val = torch.min(result[c])
                max_val = torch.max(result[c])
                if max_val > min_val:  # Avoid division by zero
                    result[c] = (result[c] - min_val) / (max_val - min_val)
            return result
        else:
            # Mean-std normalization
            img_tensor = torch.from_numpy(img)
            for c in range(img.shape[0]):
                mean = self.means[c] if self.means else img_tensor[c].mean()
                std = self.stds[c] if self.stds else img_tensor[c].std()
                if std > 0:  # Avoid division by zero
                    img_tensor[c] = (img_tensor[c] - mean) / std
            return img_tensor
        
        
if __name__ == "__main__":
    # Example usage
    zarr_path = "burned_area_dataset.zarr"
    
    dataset_set = "trainval"
    # Create DataLoader
    dataloader = get_zarr_dataloader(
        zarr_path=zarr_path,
        dataset_set=dataset_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        transform=NormalizeChannels(min_max=True),
        task_filter="classification",
        metadata_keys=["sensor", "timestamp"],
    )
    
    # Iterate through batches
    for batch in dataloader:
        # Access data based on tasks in the batch
        for task in batch['tasks']:
            images = batch[f'{task}_img']
            labels = batch[f'{task}_label']
            # Forward pass, compute loss, etc.