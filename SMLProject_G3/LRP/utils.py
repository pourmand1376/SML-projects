import sys
import pathlib

base_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, base_path.as_posix())
from converter import convert_vgg

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vgg import vgg16
from lrp import  Linear, Conv2d
from sequential    import Sequential
from maxpool       import MaxPool2d
from converter     import convert_vgg
_standard_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def get_mnist_model():
    model = Sequential(
        Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        MaxPool2d(2,2),
        nn.Flatten(),
        Linear(14*14*64, 512),
        nn.ReLU(),
        Linear(512, 10)
    )
    return model


def get_cifar10_model():
    model = Sequential(

            # Conv Layer block 1
            Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            MaxPool2d(2, 2),

            # Conv Layer block 2
            Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            MaxPool2d(2, 2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Dropout(p=0.1),
            Linear(4096, 1024),
            nn.ReLU(inplace=True),
            Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            Linear(512, 10)

        )
    return model



def get_cifar10_data(transform, batch_size=32):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)


    return trainloader, testloader




def prepare_cifar10_model(device):
    vgg = vgg16()
    # # # vgg = torchvision.models.vgg16(pretrained=True).to(device)
    checkpoint = torch.load('./models/cifar10_vgg16.tar')
    vgg.load_state_dict(checkpoint['state_dict'])
    vgg.to(device)
    vgg.eval()
    lrp_vgg = convert_vgg(vgg).to(device)

    return lrp_vgg



def prepare_cifar10_model2(device, model, model_path=(base_path / 'models' / 'cifar10_model_trained.pth').as_posix(), epochs=1, lr=1e-3, train_new=False, transform=_standard_transform):
    train_loader, test_loader = get_cifar10_data(transform)

    if os.path.exists(model_path) and not train_new:
        state_dict = torch.load(model_path, map_location=device )
        model.load_state_dict(state_dict)
        print('loaded!!!!!!!!!!!')
    else:
        model = model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
                if i%10 == 0:
                    print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
        torch.save(model.state_dict(), model_path)




def get_mnist_data(transform, batch_size=32):
    train = torchvision.datasets.MNIST((base_path / 'data').as_posix(), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torchvision.datasets.MNIST((base_path / 'data').as_posix(), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def prepare_mnist_model(device, model, model_path=(base_path / 'models' / 'mnist_model_trained.pth').as_posix(), epochs=1, lr=1e-3, train_new=False, transform=_standard_transform):
    train_loader, test_loader = get_mnist_data(transform)

    if os.path.exists(model_path) and not train_new:
        state_dict = torch.load(model_path )
        model.load_state_dict(state_dict)
    else: 
        model = model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
                if i%10 == 0: 
                    print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
        torch.save(model.state_dict(), model_path)



