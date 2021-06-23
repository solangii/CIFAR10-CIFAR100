import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from model.LeNet import LeNet
from model.MLP import MLP
from model.MobileNet import MobileNet
from model.ResNet import ResNet18
from model.VGG import VGG
from model.GoogLeNet import GoogLeNet
from model.LeNet_ablation import LeNet_1, LeNet_2

from utils import experiment_name_generator
from utils import progress_bar

from data import dataloader

total_trainloss =[]
total_valloss =[]
run_time = []

best_acc =0


def parser_args():
    parser = argparse.ArgumentParser(description='CIFAR10,100 Traning')

    # select model
    parser.add_argument('--model', default='CNN', type=str, help='select model')

    # data parameter
    parser.add_argument('--dataset', default='cifar10', type=str, help='select dataset')
    parser.add_argument('--n_train', default=20000, type=int, help='Number of train data(including val)')

    # train parameter
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--batch', default=100, type=int, help='batch size')

    #parser.add_argument('--seed', default=0, type=int, help='seed number')

    # save option
    parser.add_argument('--save', default='./checkpoint/ckpt.pth', type=str, help='save model weight')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    return config

def train(epoch, trainloader, device, model, optimizer, criterion):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    correct = 0
    total =0
    temp = []

    for idx, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (idx + 1), 100. * correct / total, correct, total))

        temp.append(train_loss / (idx + 1))
    total_trainloss.append(sum(temp) / len(temp))

def val(epoch, valloader, device, model,criterion,exp_name):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    temp = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            temp.append(test_loss / (batch_idx+1))
        total_valloss.append(sum(temp) / len(temp))


        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            name = './checkpoint/'+exp_name+'.pth'
            torch.save(state, name)
            best_acc = acc

def test(testloader, device, model, PATH):
    model.load_state_dict(torch.load(PATH), strict=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def main():
    config = parser_args()
    exp_name = experiment_name_generator(config)

    trainloader, valloader, testloader = dataloader(config)

    if config.dataset == 'cifar10' or 'fasion_mnist':
        outsize = 10
    elif config.dataset == 'cifar100':
        outsize = 100

    if config.model == 'CNN':
        if config.dataset == 'fasion_mnist':
            model = LeNet_2(outsize)
        else:
            model = LeNet(outsize)
    elif config.model == 'MLP':
        model = MLP(outsize)
    elif config.model == 'resnet':
        model = ResNet18()
    elif config.model == 'vgg':
        model = VGG('VGG19')
    elif config.model == 'googlenet':
        model = GoogLeNet()
    elif config.model == 'ablation':
        model = LeNet_1()

    model = model.to(config.device)
    print("device : ", config.device)

    if config.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Define Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config.epochs):
        s = time.time()
        train(epoch, trainloader, config.device, model, optimizer, criterion)
        val(epoch, valloader, config.device, model, criterion,exp_name)
        f = time.time()
        rt = f - s
        run_time.append(rt)


    #print("############")
    #print("train loss : ", total_trainloss )
    #print("############")
    #print("############")

    #print("val loss : ", total_valloss)
    #print("############")
    #print("############")

    print("run time : ", sum(run_time), "sec")
    #print(run_time)

    print("best val acc : ", best_acc)


    path = './checkpoint/'+exp_name+'.pth'
    test(testloader,config.device,model, path)

    #loss_plot()







if __name__ == '__main__':
    main()