import tensorflow as tf


class AlphaZeroNetwork(tf.keras.Model):
    def __init__(self, num_actions):
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
    
def create_loss_function():
    """Create the loss function for training."""
    def loss_fn(policy_targets, value_targets, policy_predictions, value_predictions):
        policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=policy_targets, logits=policy_predictions)
        )
        value_loss = tf.reduce_mean(tf.square(value_targets - value_predictions))
        return policy_loss + value_loss
    return loss_fn
    
def train_network(network, training_data, optimizer, loss_fn):
    """Train the neural network using the collected data."""
    states = tf.convert_to_tensor([data[0] for data in training_data], dtype=tf.float32)
    policy_targets = tf.convert_to_tensor([data[1] for data in training_data], dtype=tf.float32)
    value_targets = tf.convert_to_tensor([[data[2]] for data in training_data], dtype=tf.float32)

    with tf.GradientTape() as tape:
        policy_logits, value_predictions = network(states, training=True)
        loss = loss_fn(policy_targets, value_targets, policy_logits, value_predictions)

    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    return loss
