import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import params
# Import Data Loaders 
# from dataloader import *


def get_dataset(dataset, root_dir, imageSize, batchSize, workers=1):
    if dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=root_dir, download=True, train=True,
                                     transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=params.dataset_mean,
                                                    std=params.dataset_std),
                                      ]))
        test_dataset = dset.CIFAR10(root=root_dir, download=True, train=False,
                                    transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=params.dataset_mean,
                                                    std=params.dataset_std),
                                      ]))
    elif dataset == 'MNIST':
        train_dataset = dset.MNIST(root=root_dir, train=True, download=True,
                                   transform=transforms.Compose([
                                    # transforms.Resize(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        test_dataset = dset.MNIST(root=root_dir, train=False, download=True,
                                  transform=transforms.Compose([
                                    # transforms.Resize(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
    # elif dataset == 'mnistm':
    #     train_dataset = MNIST_M(root=root_dir, train=True,
    #                              transform=transforms.Compose([
    #                              transforms.Scale(imageSize),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize(
    #                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                              ]))
    #     test_dataset = MNIST_M(root=root_dir, train=False,
    #                              transform=transforms.Compose([
    #                              transforms.Scale(imageSize),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5),
    #                               (0.5, 0.5, 0.5)),
    #                              ]))
    elif dataset == 'SVHN':
        train_dataset = dset.SVHN(root=root_dir, split='train', download=True,
                                  transform=transforms.Compose([
                                            transforms.Resize((28, 28)),
                                            transforms.Grayscale(
                                                num_output_channels=1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                              mean=params.dataset_mean,
                                              std=params.dataset_std),
                                      ]))
        test_dataset = dset.SVHN(root=root_dir, split='test', download=True,
                                 transform=transforms.Compose([
                                           transforms.Resize((28, 28)),
                                           transforms.Grayscale(
                                               num_output_channels=1),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                             mean=params.dataset_mean,
                                             std=params.dataset_std),
                                      ]))

    assert train_dataset, test_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batchSize,
                                                   shuffle=True,
                                                   num_workers=int(workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batchSize,
                                                  shuffle=False,
                                                  num_workers=int(workers))
    return train_dataloader, test_dataloader