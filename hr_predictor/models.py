import tensorflow as tf


def branch_input(x_data):
    return [x_data[:, :, 0], x_data[:, :, 1], x_data[:, :, 2]]


def compute_error(y_pred, y_true):
    return (y_pred - y_true) / y_true


def get_compiled_lstm_model(input_shape):

    ts_inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(units=100)(ts_inputs)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
    model.compile(optimizer="adam",
          loss=tf.keras.losses.MeanSquaredError(),
    )

    return model


def get_compiled_dense_model(input_shape):
    input_layers = [
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Input(input_shape)
    ]

    embds = input_layers
    embds = [tf.keras.layers.Dense(50)(layer) for layer in embds]
    embds = [tf.keras.layers.Dropout(0.3)(layer) for layer in embds]

    concat_layer = tf.keras.layers.Concatenate()(embds)

    dense_layer = concat_layer
    for i in range(5):
        dense_layer = tf.keras.layers.Dense(250)(dense_layer)
        dense_layer = tf.keras.layers.Dropout(0.3)(dense_layer)

    output = tf.keras.layers.Dense(1, activation='relu')(dense_layer)

    model = tf.keras.Model(
        inputs=input_layers,
        outputs=output
    )

    model.compile(
        loss="mean_squared_error",
        optimizer='adam',
    )
    return model


def get_compiled_dense_lstm_model(input_shape):

    input_layers = [
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Input(shape=input_shape)
    ]

    embds = input_layers
    embds = [tf.keras.layers.Dense(50)(layer) for layer in embds]
    embds = [tf.keras.layers.Dropout(0.3)(layer) for layer in embds]

    concat_layer = tf.keras.layers.Concatenate()(embds)
    concat_layer = tf.keras.layers.Reshape((-1, 1))(concat_layer)

    lstm_layer = tf.keras.layers.LSTM(units=100)(concat_layer)
    lstm_layer = tf.keras.layers.Dropout(0.2)(lstm_layer)

    outputs = tf.keras.layers.Dense(1, activation='relu')(lstm_layer)

    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    model.compile(optimizer="adam",
          loss="mean_squared_error",
    )

    return model
