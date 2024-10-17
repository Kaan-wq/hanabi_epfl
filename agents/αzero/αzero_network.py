import tensorflow as tf

class AlphaZeroNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_filters=64, num_blocks=[3, 4, 6, 3]):
        super(AlphaZeroNetwork, self).__init__()
        self.num_actions = num_actions

        # Initial convolutional layer
        self.conv1 = tf.keras.layers.Conv1D(num_filters, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')

        # ResNet stages
        self.layer1 = self._make_layer(num_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_filters * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_filters * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_filters * 8, num_blocks[3], stride=2)

        # Global Average Pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')

        # Policy and Value heads
        self.policy_head = tf.keras.layers.Dense(num_actions)
        self.value_head = tf.keras.layers.Dense(1, activation='tanh')

    def _make_layer(self, filters, blocks, stride):
        downsample = None
        layers_list = []

        if stride != 1:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

        layers_list.append(ResidualBlock1D(filters, stride, downsample))

        for _ in range(1, blocks):
            layers_list.append(ResidualBlock1D(filters))

        return tf.keras.Sequential(layers_list)

    def call(self, x, training=False):
        # Initial convolutional block
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # Global average pooling
        x = self.global_avg_pool(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Policy and Value heads
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

def train_network(network, training_data, optimizer, loss_fn, batch_size=32):
    """Train the neural network using the collected data."""
    state_vectors = [data[0] for data in training_data]
    policy_targets = [data[1] for data in training_data]
    value_targets = [data[2] for data in training_data]

    states_tensor = tf.stack(state_vectors)
    policy_targets_tensor = tf.stack(policy_targets)
    value_targets_tensor = tf.expand_dims(tf.stack(value_targets), axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((states_tensor, policy_targets_tensor, value_targets_tensor))
    dataset = dataset.shuffle(buffer_size=len(training_data))
    dataset = dataset.batch(batch_size)
    
    total_loss = 0.0
    num_batches = 0

    for batch in dataset:
        states, policy_t, value_t = batch
        
        with tf.GradientTape() as tape:
            policy_logits, value_predictions = network(states, training=True)
            loss = loss_fn(policy_t, value_t, policy_logits, value_predictions)
        
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))
        
        loss_value = loss.numpy()
        total_loss += loss_value
        num_batches += 1
    
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


class ResidualBlock1D(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, downsample=None, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)

        return out
