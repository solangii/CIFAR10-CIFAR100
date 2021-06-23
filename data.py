import torch
from torch.utils.data import random_split
import torchvision
from torchvision import datasets, transforms

def dataloader(config):
    if config.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainset, valset, _ = random_split(trainset, [config.n_train // 2, config.n_train // 2, 50000 - config.n_train])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    elif config.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainset, valset, _ = random_split(trainset, [config.n_train // 2, config.n_train // 2, 50000 - config.n_train])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif config.dataset =='fasion_mnist':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        trainset, valset, _ = random_split(trainset, [config.n_train // 2, config.n_train // 2, 60000 - config.n_train])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch, shuffle=False)



    return trainloader, valloader, testloader