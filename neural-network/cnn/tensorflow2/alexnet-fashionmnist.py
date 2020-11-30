import tensorflow as tf
import argparse
import sys
sys.path.append("..") 
import mlutils.tensorflow as mlutils

def net():
    # conv: floor((input_shape - kernel_size + padding + stride) / stride)
    # input shape: 1 * 224 * 224
    return tf.keras.models.Sequential([
        # conv: 1 * 224 * 224 -> 96 * 54 * 54 
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
        # 96 * 54 * 54 -> 96 * 26 * 26
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # conv layer 2: decrease kernel size, add padding to keep input and output size same, increase channel number
        # 96 * 26 * 26 -> 256 * 26 * 26
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
        # 256 * 26 * 26 -> 256 * 12 * 12
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 3 consecutive conv layer, smaller kernel size
        # 256 * 12 * 12 -> 384 * 12 * 12
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        # 384 * 12 * 12 -> 384 * 12 * 12
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        # 384 * 12 * 12 -> 256 * 12 * 12
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        # 256 * 5 * 5
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Use the dropout layer to mitigate overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. 
        # the number of classes in Fashion-MNIST is 10
        tf.keras.layers.Dense(10)
    ])

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

    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)

    # train
    train(net, train_iter, test_iter, args.num_epochs, args.lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)