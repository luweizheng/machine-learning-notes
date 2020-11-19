import os
# disable DEBUG/INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import time

def net():
    return tf.keras.models.Sequential([
        # 卷积层1
        # padding='same'：在输入矩阵上增加padding，使得输出矩阵的大小与输入矩阵相同，输入28 * 28，输出28 * 28
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        # 6 * 28 * 28 -> 6 * 14 * 14
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        # 卷积层2
        # 6 * 14 * 14 -> 16 * 10 * 10
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        # 16 * 10 * 10 -> 16 * 5 * 5
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        # 全连接层1
        tf.keras.layers.Dense(120, activation='sigmoid'),
        # 全连接层2
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])

def load_data_fashion_mnist(batch_size, resize=None):
    """Use keras datasets module to download the Fashion-MNIST dataset and then load it into memory."""
    
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. 
    # cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

class Timer:
    """Timer class to record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class TrainCallback(tf.keras.callbacks.Callback):
    """A Callback class to log the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = Timer()
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        if epoch % 10 == 0:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'epoch {epoch + 1}: loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            if epoch == self.num_epochs -1:
                print(f'Training End: {num_examples / self.timer.avg():.1f} examples/sec on '
                      f'{str(self.device_name)}')

def train(net_fn, train_iter, test_iter, num_epochs, lr, device=try_gpu()):
    """Train a model with a GPU."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs, device_name)
    history = net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net

def main():

    batch_size = 256
    lr, num_epochs = 0.9, 100
    
    # load data
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # train
    train(net, train_iter, test_iter, num_epochs, lr)

if __name__ == '__main__':
    main()