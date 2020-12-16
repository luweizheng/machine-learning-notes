import torch
from torch import nn
from torch.nn import functional as F
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

class InceptionBlock(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)

def googlenet():
    # input shape: 1 * 96 * 96
    # convolutional and max pooling layer 
    # 1 * 96 * 96 -> 64 * 24 * 24
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # convolutional and max pooling layer 
    # 64 * 24 * 24 -> 192 * 12 * 12
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 2 InceptionBlock
    # 192 * 12 * 12 -> 480 * 6 * 6
    b3 = nn.Sequential(InceptionBlock(192, 64, (96, 128), (16, 32), 32),
                   InceptionBlock(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 5 InceptionBlock
    # 480 * 6 * 6 -> 832 * 3 * 3
    b4 = nn.Sequential(InceptionBlock(480, 192, (96, 208), (16, 48), 64),
                   InceptionBlock(512, 160, (112, 224), (24, 64), 64),
                   InceptionBlock(512, 128, (128, 256), (24, 64), 64),
                   InceptionBlock(512, 112, (144, 288), (32, 64), 64),
                   InceptionBlock(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 2 InceptionBlock and max pooling layer
    # 832 * 3 * 3 -> 1024 * 1
    b5 = nn.Sequential(InceptionBlock(832, 256, (160, 320), (32, 128), 128),
                   InceptionBlock(832, 384, (192, 384), (48, 128), 128),
                   # AdaptiveMaxPool2d convert the matrix into 1 * 1 scalar
                   nn.AdaptiveMaxPool2d((1,1)),
                   nn.Flatten())
    # final linear layer, for classification label number
    # 1024 -> 10
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    return net

def main(args):
    net = googlenet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    # resize into 96 * 96
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=96)
    # train
    mlutils.train(net, train_iter, test_iter, args.batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)
    