# utils.py

import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data_loader(X, y, batch_size=64):
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
