import os
# disable DEBUG/INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import sys
sys.path.append("..") 
import mlutils.tensorflow as mlutils

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

def train(net_fn, train_iter, test_iter, num_epochs, lr, device=mlutils.try_gpu()):
    """Train a model with a GPU."""
    device_name = device._device_name
    print("training on", device_name)
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = mlutils.TrainCallback(net, train_iter, test_iter, num_epochs, device_name)
    history = net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net

def main(args):
    # The original VGG network has 5 convolutional blocks.
    # The first two blocks have one convolutional layer.
    # The latter three blocks contain two convolutional layers.
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    # The parameters of VGG-11 are big, use a ratio to reduce the network size by dividing a ratio on the output channel number.
    ratio = args.ratio
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = lambda: vgg(small_conv_arch)

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    train(net, train_iter, test_iter, args.num_epochs, args.lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ratio', type=float, default=4, help='ratio to reduce the network parameters size')
    args = parser.parse_args()
    main(args)