import torch
from torch import nn
from torch.nn import functional as F
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, in_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """return multiple resnet blocks"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def resnet():
    """return resnet"""
    b1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(in_channels=64, num_channels=64, num_residuals=2, first_block=True))
    b3 = nn.Sequential(*resnet_block(in_channels=64, num_channels=128, num_residuals=2))
    b4 = nn.Sequential(*resnet_block(in_channels=128, num_channels=256, num_residuals=2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

    return net

def train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device=mlutils.try_gpu()):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    timer = mlutils.Timer()
    # in one epoch, it will iterate all training samples
    for epoch in range(num_epochs):
        # Accumulator has 3 parameters: (loss, train_acc, number_of_images_processed)
        metric = mlutils.Accumulator(3)
        # all training samples will be splited into batch_size
        for X, y in train_iter:
            timer.start()
            # set the network in training mode
            net.train()
            # move data to device (gpu)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # all the following metrics will be accumulated into variable `metric`
                metric.add(l * X.shape[0], mlutils.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            # metric[0] = l * X.shape[0], metric[2] = X.shape[0]
            train_l = metric[0]/metric[2]
            # metric[1] = number of correct predictions, metric[2] = X.shape[0]
            train_acc = metric[1]/metric[2]
        test_acc = mlutils.evaluate_accuracy_gpu(net, test_iter)
        if epoch % 1 == 0:
            print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f'total training time {timer.sum():.2f}, {metric[2] * num_epochs / timer.sum():.1f} images/sec ' f'on {str(device)}')

def main(args):
    net = resnet()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    train(net, train_iter, test_iter, args.batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)
