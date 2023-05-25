import torch


class MstarDataset(torch.utils.data.Dataset):
    def __init__(self, path, name='soc', is_train=False, transform=None):
        self.is_train = is_train
        self.name = name

        self.images = []
        self.labels = []
        self.serial_number = []

        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        _label = self.labels[idx]
        _serial_number = self.serial_number[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label, _serial_number
