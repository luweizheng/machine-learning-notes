import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import time
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def load_data_cifar10(batch_size, resize=None, root='~/Datasets/CIFAR10'):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(
        root=root, train=True, transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(
        root=root, train=False, transform=transform_test)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.nprocs, rank=local_rank)

    torch.cuda.set_device(local_rank)
    net = ResNet18()

    net.cuda(local_rank)

    args.batch_size = int(args.batch_size / args.nprocs)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    loss = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # prepare data
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(
        root=args.data, train=True, transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(
        root=args.data, train=False, transform=transform_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar_train)
    train_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)

    # test loader does not have to follow distributed sampling strategy
    test_loader = torch.utils.data.DataLoader(dataset=cifar_test, batch_size=128, shuffle=False, num_workers=8)

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        
        train(net, train_loader, test_loader, loss, epoch, optimizer, local_rank, args)
        acc = validate(net, test_loader, loss, local_rank, args)

        print(f"Epoch: {epoch + 1}, Accuracy: {acc}")

def train(net, train_iter, test_iter, loss, epoch, optimizer, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # set the network in training mode
    net.train()

    for X, y in train_iter:
        # move data to device (gpu)
        X = X.cuda(local_rank, non_blocking=True)
        y = y.cuda(local_rank, non_blocking=True)

        y_hat = net(X)

        l = loss(y_hat, y)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(l, args.nprocs)
        losses.update(reduced_loss.item(), X.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

def validate(net, test_iter, loss, local_rank, args):

    correct = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_iter):
            X = X.cuda(local_rank, non_blocking=True)
            y = y.cuda(local_rank, non_blocking=True)

            # compute output
            y_hat = net(X)

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total

    return accuracy

def main(args):

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 Distributed Training')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:20000')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--data', type=str, default='~/Datasets/CIFAR10', help='CIFAR10 dataset path')
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main(args)