import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data: list, size: int):
        self.size = size
        self.x = torch.unsqueeze(torch.tensor(data, dtype=torch.float32), dim=-1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx]
