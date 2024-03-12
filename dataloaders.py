from torch.utils.data import Dataset, DataLoader
from MatrixVectorizer import *
import pandas as pd
import torch

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


lr_data_path = './data/lr_train.csv'
hr_data_path = './data/hr_train.csv'

lr_data = pd.read_csv(lr_data_path, delimiter=',').to_numpy()
hr_data = pd.read_csv(hr_data_path, delimiter=',').to_numpy()
lr_data[lr_data < 0] = 0
np.nan_to_num(lr_data, copy=False)
hr_data[hr_data < 0] = 0
np.nan_to_num(hr_data, copy=False)

lr_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 160) for row in lr_data]))
hr_data_vectorized = torch.tensor(np.array([MatrixVectorizer.anti_vectorize(row, 268) for row in hr_data]))



brain_dataset = NoisyDataset(lr_data_vectorized, hr_data_vectorized, noise_level=0.5)