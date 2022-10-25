import tensorflow as tf

class DeepQNetwork(tf.keras.model):
    def __init__(self, action_size, state_size):

        self.action_size = action_size
        self.state_size = state_size

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", input_shape=self.state_size)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.action_size, activation="linear")

    def call(self, X):
        l1 = self.conv1(X)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.flatten(l3)
        l5 = self.dense1(l4)
        output = self.dense2(l5)

        return output