from __future__ import print_function

import glob
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

NET_PATH = 'trained_nets/mnist_net_%s.pth' % time.strftime('%Y-%m-%d_%H-%M')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(6 * 6 * 64, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        # input 32x32x1 (2 padded 28x28 images 1-channel) -> 28x28x32 (Conv1) -> 14x14x32 (Max pool (2,2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 14x14x32 -> 12x12x64 (Conv2) -> 6x6x64 (Max pool (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten features
        x = x.view(-1, 6 * 6 * 64)
        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(loader, nepoch=10, save=False):
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
                running_loss = 0.0
    if save:
        torch.save(net.state_dict(), NET_PATH)


def test(loader, load=True, load_path=None):
    net.eval()
    if load:
        net.load_state_dict(torch.load(load_path))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %.3f' % (100 * correct / total))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4,
                                          pin_memory=True if torch.cuda.is_available() else False)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4,
                                         pin_memory=True if torch.cuda.is_available() else False)


def dataset_wide_mean_std(dataset):
    return dataset.train_data.float().mean() / 255, dataset.train_data.float().std() / 255


if __name__ == '__main__':
    # Compute dataset-wide mean and std
    # mean, std = dataset_wide_mean_std(trainset)
    # print(mean, std)

    # mean = 0.
    # std = 0.
    # for images, _ in trainloader:
    #     batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #
    # mean /= len(trainloader.dataset)
    # std /= len(trainloader.dataset)

    train(trainloader, save=False)
    # list_of_file_paths = glob.glob('trained_nets/*.pth')
    # latest_net_path = max(list_of_file_paths, key=os.path.getctime)
    test(testloader, load=False)
