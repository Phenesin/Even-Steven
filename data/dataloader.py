from torch.utils.data import DataLoader
from .dataset import QuickDrawDataset

import numpy as np
rng = np.random.default_rng(seed = 42)


def get_dataloaders(
        data_dir,
        classes,
        samples_per_class,
        batch_size,
        train_transform = None,
        test_transform = None,
        train_ratio = 0.8,
        val_ratio = 0.1
    ):
        dataset = QuickDrawDataset(
                data_dir = data_dir,
                classes = classes,
                samples_per_class = samples_per_class,
                transform = None
        )

        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        indices = rng.permutation(total_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        test_indices = indices[train_size + val_size:]
        
        train_dataset = QuickDrawDataset(
                data_dir = data_dir,
                classes = classes,
                samples_per_class = samples_per_class,transform = train_transform,
                indices = train_indices)

        test_dataset = QuickDrawDataset(
                data_dir = data_dir,
                classes = classes,
                samples_per_class = samples_per_class,transform = test_transform,
                indices = test_indices)

        val_dataset = QuickDrawDataset(
                data_dir = data_dir,
                classes = classes,
                samples_per_class = samples_per_class,transform = test_transform,
                indices = val_indices)


        train_loader = DataLoader(
                train_dataset,
                batch_size = batch_size,
                shuffle = True
        )

        val_loader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = False
        )

        test_loader = DataLoader(
                test_dataset,
                batch_size = batch_size,
                shuffle = False
        )

        return train_loader, val_loader, test_loader