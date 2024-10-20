import tensorflow as tf
import numpy as np


def AlphaZeroNetwork(num_actions, obs_shape, num_filters=64, num_blocks=[2, 2, 2, 2]):
    inputs = tf.keras.layers.Input(shape=(1, obs_shape, 1))

    # Initial convolutional block
    x = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(1, 7),
        strides=(1, 2),
        padding="same",
        use_bias=False,
        name="conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(1, 3), strides=(1, 2), padding="same", name="maxpool"
    )(x)

    # ResNet stages
    x = _make_layer(x, num_filters, num_blocks[0], stride=1, stage=1)
    x = _make_layer(x, num_filters * 2, num_blocks[1], stride=2, stage=2)
    x = _make_layer(x, num_filters * 4, num_blocks[2], stride=2, stage=3)
    x = _make_layer(x, num_filters * 8, num_blocks[3], stride=2, stage=4)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

    # Fully connected layers
    x = tf.keras.layers.Dense(512, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc2")(x)

    # Policy and Value heads
    policy_logits = tf.keras.layers.Dense(num_actions, name="policy_logits")(x)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(x)

    # Create model
    model = tf.keras.Model(
        inputs=inputs, outputs=[policy_logits, value], name="AlphaZeroNetwork"
    )

    return model


def _make_layer(input_tensor, filters, blocks, stride=1, stage=1):
    x = input_tensor
    for block in range(blocks):
        block_stride = stride if block == 0 else 1
        x = residual_block(x, filters, stride=block_stride, block=block, stage=stage)
    return x


def residual_block(input_tensor, filters, stride=1, block=0, stage=1):
    conv_name_base = f"res{stage}_block{block}_branch"
    bn_name_base = f"bn{stage}_block{block}_branch"

    # Main path
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 3),
        strides=(1, stride),
        padding="same",
        use_bias=False,
        name=conv_name_base + "_2a",
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + "_2a")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 3),
        padding="same",
        use_bias=False,
        name=conv_name_base + "_2b",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + "_2b")(x)

    # Shortcut path
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(1, stride),
            padding="same",
            use_bias=False,
            name=conv_name_base + "_1",
        )(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + "_1")(
            shortcut
        )
    else:
        shortcut = input_tensor

    # Combine paths
    x = tf.keras.layers.Add(name=f"res{stage}_block{block}_add")([x, shortcut])
    x = tf.keras.layers.Activation("relu", name=f"res{stage}_block{block}_out")(x)
    return x


def prepare_data(training_data, batch_size=16):
    while True:
        np.random.shuffle(training_data)
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i + batch_size]
            state_vectors = [data[0] for data in batch_data]
            policy_targets = [data[1] for data in batch_data]
            value_targets = [data[2] for data in batch_data]

            states_tensor = tf.stack(state_vectors)
            states_tensor = tf.reshape(states_tensor, [-1, 1, states_tensor.shape[1], 1])

            policy_targets_tensor = tf.stack(policy_targets)
            value_targets_tensor = tf.expand_dims(tf.stack(value_targets), axis=-1)

            yield states_tensor, {
                "policy_logits": policy_targets_tensor,
                "value": value_targets_tensor,
            }
