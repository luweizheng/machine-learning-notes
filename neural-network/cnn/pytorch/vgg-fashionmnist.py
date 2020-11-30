import torch
from torch import nn
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

def vgg_block(num_convs, in_channels, out_channels):
    '''
    Returns a block of convolutional neural network

        Parameters:
            num_convs (int): number of convolutional layers this block has
            in_channels (int): input channel number of this block
            out_channels (int): output channel number of this block

        Returns:
            a nn.Sequential network: 
    '''
    layers=[]
    for _ in range(num_convs):
        # (input_shape - 3 + 2 + 1) / 1 = input_shape
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # (input_shape - 2 + 2) / 2 = input_shape / 2
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    '''
    Returns the VGG network

        Parameters:
            conv_arch ([(int, int)]): a list which contains vgg block info.
                the first element is the number of convolution layers this block have.
                the latter element is the output channel number of this block.

        Returns:
            a nn.Sequential network: 
    '''
    # The convolutional part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

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
    # The original VGG network has 5 convolutional blocks.
    # The first two blocks have one convolutional layer.
    # The latter three blocks contain two convolutional layers.
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    # The parameters of VGG-11 are big, use a ratio to reduce the network size by dividing a ratio on the output channel number.
    ratio = args.ratio
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    train(net, train_iter, test_iter, batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ratio', type=float, default=4, help='ratio to reduce the network parameters size')
    args = parser.parse_args()
    main(args)