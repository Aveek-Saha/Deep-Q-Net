import tensorflow as tf

# class DeepQNetwork(tf.keras.Model):
#     def __init__(self, action_size, frame_height, frame_width, agent_history_length):
#         super(DeepQNetwork, self).__init__()
#         self.action_size = action_size
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#         self.agent_history_length = agent_history_length

#         self.lam = tf.keras.layers.Lambda(lambda x: x / 255)
#         self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)
#         self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)
#         self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation='relu')
#         self.dense2 = tf.keras.layers.Dense(self.action_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="linear")

#     def call(self, X):
#         norm = self.lam(X)
#         l1 = self.conv1(norm)
#         l2 = self.conv2(l1)
#         l3 = self.conv3(l2)
#         l4 = self.flatten(l3)
#         l5 = self.dense1(l4)
#         output = self.dense2(l5)

#         return output

def DeepQNetwork(learning_rate, action_size, frame_height, frame_width, agent_history_length):
    model_input = tf.keras.layers.Input(shape=(frame_height, frame_width, agent_history_length))
    x = tf.keras.layers.Lambda(lambda x: x / 255)(model_input) 

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation='relu')(x)

    x = tf.keras.layers.Dense(action_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="linear")(x)

    model = tf.keras.Model(model_input, x)
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

def DuellingDQN(learning_rate, action_size, frame_height, frame_width, agent_history_length):
    model_input = tf.keras.layers.Input(shape=(frame_height, frame_width, agent_history_length))
    x = tf.keras.layers.Lambda(lambda x: x / 255)(model_input) 

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=7, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), activation="relu", use_bias=False)(x)

    val_stream, adv_stream = tf.keras.layers.Lambda(lambda i: tf.split(i, 2, 3))(x)

    val_stream = tf.keras.layers.Flatten()(val_stream)
    val = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(val_stream)
    adv_stream = tf.keras.layers.Flatten()(adv_stream)
    adv = tf.keras.layers.Dense(action_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(adv_stream)

    q = tf.keras.layers.Add()([val, tf.keras.layers.Subtract()([adv, tf.keras.layers.Lambda(lambda i: tf.reduce_mean(i, axis=1, keepdims=True))(adv)])])

    model = tf.keras.Model(model_input, q)
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model