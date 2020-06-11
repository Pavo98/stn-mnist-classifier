import torch.utils.data as data
import torch.optim as optim

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from stn_module import *


class RandomRectangleDataset:
    def __init__(self, n=1000, shape=(1, 40, 40)):
        self.n = n
        self.shape = shape
        self.target = torch.zeros(shape)
        self.target[:, 5:-5, 5:-5] = torch.linspace(0.5, 1.0, shape[1] - 10).view(1, shape[1] - 10, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randint(0, self.shape[2] - 10, (1,)).to(torch.int).item()
        y = torch.randint(0, self.shape[1] - 10, (1,)).to(torch.int).item()
        h = torch.randint(2, self.shape[1] - y, (1,)).to(torch.int).item()
        w = torch.randint(2, self.shape[2] - x, (1,)).to(torch.int).item()

        img = torch.zeros(self.shape)
        img[:, y:y + h, x:x + h] = torch.linspace(0.5, 1.0, h).view(1, h, 1)

        return img, self.target


def train(net, dev, trainloader, testloader, nepoch=5, lr=1e-3):
    net.train()

    opt = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(nepoch):
        net.train()
        train_loss = 0.
        for x, y in trainloader:
            opt.zero_grad()

            x, y = x.to(dev), y.to(dev)
            out = net(x)

            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        net.eval()
        test_loss = 0.
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(dev), y.to(dev)
                out = net(x)
                loss = F.mse_loss(out, x)
                test_loss += loss.item()

        if epoch % 10 == 9:
            print(epoch + 1, train_loss / len(trainloader), test_loss / len(testloader))


def test_stn(simple=False):
    imgshape = (1, 40, 40)
    trainset = RandomRectangleDataset(shape=imgshape)
    testset = RandomRectangleDataset(n=100, shape=imgshape)

    trainloader = data.DataLoader(trainset, batch_size=32, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=32, shuffle=False)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stn_net = {
        False: STNtps,
        True: STNSimple,
    }[simple]
    net = stn_net(imgshape, (6, 6)).to(dev)
    train(net, dev, trainloader, testloader, nepoch=40, lr=1e-4)

    x, y = next(iter(testloader))
    with torch.no_grad():
        net.eval()
        xx = net(x.to(dev))

    gx = make_grid(x, padding=2, pad_value=1)
    gxx = make_grid(xx, padding=2, pad_value=1)
    gy = make_grid(y, padding=2, pad_value=1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(gx.permute(1, 2, 0).numpy())
    axs[0].set_title('Input')
    axs[1].imshow(gxx.cpu().permute(1, 2, 0).numpy())
    axs[1].set_title('Output')
    axs[2].imshow(gy.permute(1, 2, 0).numpy())
    axs[2].set_title('Target')
    fig.show()


if __name__ == '__main__':
    test_stn(simple=True)
    test_stn(simple=False)
