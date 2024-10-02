from torch.utils.data import Dataset
import torch


class PairDataset(Dataset):
    def __init__(self, data, index):
        super(PairDataset, self).__init__()
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        idx1, idx2, label = self.index[index]
        return torch.tensor(self.data[idx1]['embedding']), torch.tensor(self.data[idx2]['embedding']), torch.tensor(label)
