from __future__ import print_function

import glob
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import mnist_classifier as mnist_cls
import stn_module
from helpers import conf_matrix_to_figure

if __name__ == '__main__':
    NET_PATH = 'trained_nets/stn_mnist_net_%s.pth' % time.strftime('%Y-%m-%d_%H-%M')


class STNClassifierNet(nn.Module):

    def __init__(self):
        super(STNClassifierNet, self).__init__()

        self.stn = stn_module.STN(outshape=(1, 28, 28))
        self.cls = mnist_cls.CNN()

    def forward(self, x):
        x = self.stn(x)
        x = self.cls(x)
        return x


def train(net, loader, nepoch=10, save=False):
    net.train()
    for epoch in range(nepoch):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 60 == 59:
                print('[%d, %3d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / 60 * 100))
                writer.add_scalar('training loss',
                                  running_loss / 60,
                                  global_step=epoch * len(loader) + i)
                running_loss = 0.0
    writer.close()
    if save:
        torch.save(net.state_dict(), NET_PATH)


def test(net, loader, load=True, load_path=None):
    if load:
        net.load_state_dict(torch.load(load_path))
    net.eval()
    correct = 0
    total = 0
    conf_matrix = torch.zeros((10, 10), dtype=torch.int64, requires_grad=False)
    class_labels = []
    class_probs = []
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            _, predicted = torch.max(outputs.data, 1)
            class_probs.append(class_probs_batch)
            class_labels.append(data[1])
            indices = 10 * data[1] + predicted.cpu()
            conf_matrix += torch.bincount(indices, minlength=10 ** 2).reshape(10, 10)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_labels = torch.cat(class_labels)
    for i, class_name in enumerate(testset.classes):
        writer.add_pr_curve(class_name,
                            test_labels == i,
                            test_probs[:, i])
    writer.add_figure('confusion matrix',
                      conf_matrix_to_figure(conf_matrix))
    writer.close()
    print('Accuracy on test set: %.3f' % (100 * correct / total))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = STNClassifierNet()
    net.to(device)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    rotate_degrees = 60
    # random.seed(0)

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=rotate_degrees, fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4,
                                              pin_memory=True if torch.cuda.is_available() else False)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4,
                                             pin_memory=True if torch.cuda.is_available() else False)

    # mean, std = mnist_cls.dataset_wide_mean_std(trainset)

    writer = SummaryWriter('runs/stn_net')
    dataiter = iter(trainloader)
    images, _ = dataiter.next()
    images = images.view(-1, 28, 28).mul_(0.3801).add_(0.1307).view(-1, 1, 28, 28)  # Unnormalize
    img_grid = torchvision.utils.make_grid(images[:6, ...], nrow=3)
    writer.add_graph(net, images.to(device))
    writer.add_image('random_rotate_{}-degrees'.format(rotate_degrees), img_grid)
    writer.close()

    train(net, trainloader, nepoch=100, save=True)
    list_of_file_paths = glob.glob('trained_nets/stn_*.pth')
    latest_net_path = max(list_of_file_paths, key=os.path.getctime)
    test(net, testloader, load=True, load_path=latest_net_path)
