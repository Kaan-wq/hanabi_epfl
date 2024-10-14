import tensorflow as tf

class AlphaZeroNetwork(tf.keras.Model):
    def __init__(self, observation_size, num_actions):
        super(AlphaZeroNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.policy_head = tf.keras.layers.Dense(num_actions)
        self.value_head = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
