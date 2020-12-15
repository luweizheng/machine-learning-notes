import os
import time
import torch
import torchvision
from torch import nn, optim
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # convolution layer will change input shape into: floor((input_shape - kernel_size + padding + stride) / stride)
        # input shape: 1 * 224 * 224
        # convolution part
        self.conv = nn.Sequential(
            # conv layer 1
            # floor((224 - 11 + 2 + 4) / 4) = floor(54.75) = 54
            # conv: 1 * 224 * 224 -> 96 * 54 * 54 
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            # floor((54 - 3 + 2) / 2) = floor(26.5) = 26
            # 96 * 54 * 54 -> 96 * 26 * 26
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # conv layer 2: decrease kernel size, add padding to keep input and output size same, increase channel number
            # floor((26 - 5 + 4 + 1) / 1) = 26
            # 96 * 26 * 26 -> 256 * 26 * 26
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            # floor((26 - 3 + 2) / 2) = 12
            # 256 * 26 * 26 -> 256 * 12 * 12
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3 consecutive conv layer, smaller kernel size
            # floor((12 - 3 + 2 + 1) / 1) = 12
            # 256 * 12 * 12 -> 384 * 12 * 12
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # 384 * 12 * 12 -> 384 * 12 * 12
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # 384 * 12 * 12 -> 256 * 12 * 12
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # floor((12 - 3 + 2) / 2) = 5
            # 256 * 5 * 5
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # fully connect part 
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            # Use the dropout layer to mitigate overfitting
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # Output layer. 
            # the number of classes in Fashion-MNIST is 10
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def main(args):
    net = AlexNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    mlutils.train(net, train_iter, test_iter, args.batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)