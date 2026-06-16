from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class QuickDrawDataset(Dataset):
    def __init__(
            self,
            data_dir,
            classes,
            samples_per_class = None,
            transform = None,
            indices = None
    ):
        self.transform = transform

        images = []
        labels = []
        self.class_to_idx = {
            name : idx for idx, name in enumerate(classes)
        }
        self.idx_to_class = {
            idx : name for name, idx in self.class_to_idx.items()
        }

        for class_name in classes:
            label = self.class_to_idx[class_name]

            file_path = Path(data_dir)/f"{class_name}.npy"
            data = np.load(file_path)

            if samples_per_class is not None:
                data = data[:samples_per_class]
            
            images.append(data)
            labels.append(
                np.full(len(data), label, dtype = np.int64)
            )
        self.images = np.concatenate(images, axis = 0)
        self.labels = np.concatenate(labels, axis = 0)
        
        if indices is not None:
            self.images = self.images[indices]
            self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = torch.tensor(self.labels[idx], dtype = torch.long)


        image = image.reshape(28, 28)

        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label