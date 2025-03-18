import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import joblib
from PIL import Image
import json
import os

# Pytorch build-in MNIST dataset.
def get_mnist_loader(
        root='../data',
        train=True,
        batch_size=64,
        transform=T.ToTensor(),
        num_workers=4,
        pin_memory=True,
        fashion=False
):
    """
    :param root:
    :param train:
    :param batch_size:
    :param transform:
    :param num_workers:
    :param pin_memory:
    :param fashion:
    :return: Dataloader.
    """
    dataset = (FashionMNIST if fashion else MNIST)(root=root, train=train, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataloader
    




class Functaset(Dataset):
    def __init__(self, pkl_file):
        super(Functaset, self).__init__()
        self.functaset = joblib.load(pkl_file)

    def __getitem__(self, item):
        pair = self.functaset[item]
        modul = torch.tensor(pair['modul'])
        label = torch.tensor(pair['label'])
        return modul, label

    def __len__(self):
        return len(self.functaset)


def collate_fn(data):
    """
    :param data: is a list of tuples with (modul, label).
    :return: data batch.
    """
    moduls, labels = zip(*data)
    return torch.stack(moduls), torch.stack(labels)



# Get the functa version of MNIST.
def get_mnist_functa(
        data_dir=None,
        batch_size=256,
        mode='train',
        num_workers=4,
        pin_memory=True
):
    """
    :param data_dir:
    :param batch_size:
    :param mode:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    assert mode in ['train', 'val', 'test']
    if data_dir is None:
        data_dir = f'./functaset/mnist_{mode}.pkl'
    functaset = Functaset(data_dir)
    shuffle = mode == 'train'
    return DataLoader(
        functaset, batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn,
    )
