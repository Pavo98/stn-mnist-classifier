import argparse
import glob
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import mnist_classifier as mnist_cls
import stn_mnist as stn_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--nepoch', type=int, default=10)
parser.add_argument('--test-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=50)
parser.add_argument('--model', required=True, choices=['no_stn', 'stn_tps', 'stn_affine'])
parser.add_argument('--grid-size', type=int, default=10)
parser.add_argument('--angle', type=int, default=60)
parser.add_argument('--trans', type=float, default=0.2)
args = parser.parse_args()
args.ctrl_shape = (args.grid_size, args.grid_size)
args.translate = (args.trans, args.trans)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'no_stn':
    model = mnist_cls.CNN()
    model_cls = mnist_cls
else:
    model = stn_mnist.STNClassifierNet(args)
    model_cls = stn_mnist
model.to(device)
model_cls.device = device
model_cls.optimizer = optim.Adam(model.parameters())
model_cls.cross_entropy_loss = nn.CrossEntropyLoss()

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_transform = transforms.Compose([
    transforms.RandomAffine(degrees=args.angle, translate=args.translate),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True if torch.cuda.is_available() else False)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True if torch.cuda.is_available() else False)
model_cls.testset = testset

# start_time = time.strftime('%Y-%m-%d_%H-%M')
# run_path = '%s_net_%s' % (args.model, start_time) \
#            + '_grid%d' % args.grid_size if args.model == 'stn_tps' else '' \
#            + '_angle%d_trans%.2f' % (args.angle, args.trans)

# writer = SummaryWriter('runs/' + run_path)
# model_cls.writer = writer
# dataiter = iter(trainloader)
# images, _ = dataiter.next()
# images = images.view(-1, 28, 28).mul_(0.3801).add_(0.1307).view(-1, 1, 28, 28)  # Unnormalize
# img_grid = torchvision.utils.make_grid(images[:9, ...], nrow=3)
# writer.add_graph(model, images.to(device))
# writer.add_image('examples', img_grid)
# writer.close()

# for epoch in range(1, args.nepoch + 1):
#     model_cls.train(model, trainloader, 1, global_epoch=epoch)
#     if epoch % args.test_interval == 0:
#         model_cls.test(model, testloader, load=False, global_epoch=epoch)
#     if epoch % args.save_interval == 0:
#         save_path = 'trained_nets/' + run_path + '_epoch%03d.pth' % epoch
#         torch.save(model.state_dict(), save_path)

model_paths = glob.glob('trained_nets/%s*.pth' % args.model)
pattern = re.compile(r'trained_nets/(.*?)_epoch(\d{3}).pth')
for path in sorted(model_paths):
    match = pattern.search(path)
    if not match:
        continue
    if args.model == 'stn_tps' and not match.group(1).endswith(str(args.grid_size)):
        continue
    run_path = match.group(1) + '_angle%d_trans%.2f' % (args.angle, args.trans)
    writer = SummaryWriter('runs/' + run_path)
    model_cls.writer = writer
    epoch = int(match.group(2))
    model_cls.test(model, testloader, load=True, load_path=path, global_epoch=epoch)
