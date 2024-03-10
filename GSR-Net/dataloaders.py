from torch.utils.data import Dataset, DataLoader
from MatrixVectorizer import *
import pandas as pd
import torch

TRAIN_VAL_SPLIT = 0.8

lr_data_path = '../data/lr_train.csv'
hr_data_path = '../data/hr_train.csv'

lr_data = pd.read_csv(lr_data_path, delimiter=',').to_numpy()
hr_data = pd.read_csv(hr_data_path, delimiter=',').to_numpy()
lr_data[lr_data < 0] = 0
np.nan_to_num(lr_data, copy=False)
hr_data[hr_data < 0] = 0
np.nan_to_num(hr_data, copy=False)

# Split the data into training and validation sets
split = int(lr_data.shape[0] * TRAIN_VAL_SPLIT)
lr_train_data = lr_data[:split]
hr_train_data = hr_data[:split]
lr_val_data = lr_data[split:]
hr_val_data = hr_data[split:]

lr_train_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 160) for row in lr_train_data]))
hr_train_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 268) for row in hr_train_data]))
lr_val_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 160) for row in lr_val_data]))
hr_val_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 268) for row in hr_val_data]))


class NoisyDataset(Dataset):
    def __init__(self, lr_data, hr_data, noise_level=0.01):
        """
        lr_data: Low resolution data (torch.tensor)
        hr_data: High resolution data (torch.tensor)
        noise_level: Standard deviation of Gaussian noise to be added
        """
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.noise_level = noise_level

    def __len__(self):
        return len(self.lr_data)

    def __getitem__(self, idx):
        lr_sample = self.lr_data[idx]
        hr_sample = self.hr_data[idx]

        # Adding Gaussian noise
        noise = torch.randn(lr_sample.size()) * self.noise_level
        noisy_lr_sample = lr_sample + noise

        # Clipping to ensure values are between 0 and 1
        noisy_lr_sample = torch.clamp(noisy_lr_sample, 0, 1)

        return noisy_lr_sample, hr_sample

train_data = NoisyDataset(lr_train_data_vectorized, hr_train_data_vectorized, noise_level=0.5)
val_data = NoisyDataset(lr_val_data_vectorized, hr_val_data_vectorized, noise_level=0.5)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
