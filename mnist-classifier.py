import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms


NET_PATH = './mnist_net.pth'


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


def train(loader, nepoch=10):
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
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(net.state_dict(), NET_PATH)


def test(loader):
    net.load_state_dict(torch.load(NET_PATH))
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

if __name__ == '__main__':
    # train(trainloader)
    test(testloader)
