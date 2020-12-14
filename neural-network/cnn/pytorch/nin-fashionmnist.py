import torch
from torch import nn
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

def nin():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # Transform the four-dimensional output into two-dimensional output with a
        # shape of (batch size, 10)
        nn.Flatten())

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
    net = nin()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    train(net, train_iter, test_iter, args.batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ratio', type=float, default=4, help='ratio to reduce the network parameters size')
    args = parser.parse_args()
    main(args)