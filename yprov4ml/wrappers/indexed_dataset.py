
from torch.utils.data import Dataset

class IndexedDatasetWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        batch = self.dataset[index]
        return index, batch