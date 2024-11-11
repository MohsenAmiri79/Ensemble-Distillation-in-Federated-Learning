import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np


class NoLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data


def data_loaders(conf={'num_clients': 10,
                       'server_clients_split': 0.3,
                       'batch_size': 64,
                       'TESTRUN': False}):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    num_samples = len(train_dataset)
    num_tests = len(test_dataset)

    # TEST RUNS ONLY!!!
    if conf['TESTRUN']:
        num_samples = num_samples // 10
        num_tests = num_tests // 10

        train_indices = np.random.choice(
            len(train_dataset), num_samples, replace=False)
        test_indices = np.random.choice(
            len(test_dataset), num_tests, replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=conf['batch_size'], shuffle=False)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    server_split = int(conf['server_clients_split'] * num_samples)
    remaining_split = num_samples - server_split

    server_indices = indices[:server_split]

    remaining_indices = indices[server_split:]

    scale = remaining_split // 100
    partition_sizes = np.random.normal(
        loc=remaining_split/conf['num_clients'], scale=scale,
        size=conf['num_clients']).astype(int)
    partition_sizes = np.round(
        partition_sizes / partition_sizes.sum() * remaining_split).astype(int)
    partition_sizes[-1] = remaining_split - partition_sizes[:-1].sum()

    current_index = 0
    partitions = []
    for size in partition_sizes:
        partitions.append(
            remaining_indices[current_index:current_index + size])
        current_index += size

    clients_dataloaders = [DataLoader(Subset(train_dataset, partition),
                                      batch_size=conf['batch_size'],
                                      shuffle=True)
                           for partition in partitions]

    server_dataloader = DataLoader(NoLabelsDataset(Subset(train_dataset,
                                                          server_indices)),
                                   batch_size=conf['batch_size'], shuffle=True)

    return server_dataloader, clients_dataloaders, test_loader


def create_pseudo_loader(server_dataloader, averaged_logits, conf):
    server_data = []
    for data in server_dataloader:
        server_data.append(data)
    server_data = torch.cat(server_data)

    combined_dataset = TensorDataset(server_data, averaged_logits)
    combined_dataloader = DataLoader(
        combined_dataset, batch_size=conf['batch_size'], shuffle=True)

    return combined_dataloader
