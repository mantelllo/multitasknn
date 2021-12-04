import tensorflow as tf


def train_model(model, X, y, task2_loss_multiplier=1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    loss_y1 = loss_y1_maker(task2_loss_multiplier)
    loss_y2 = loss_y2_maker(task2_loss_multiplier)
    model.compile(optimizer=optimizer, loss={'y1': loss_y1, 'y2': loss_y2}, run_eagerly=True)
    model.fit(X, y, batch_size=64, epochs=2)

    optimizer.lr = 1e-3
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
    return model.fit(X, {'y1':y,'y2':y}, batch_size=64, epochs=50, validation_split=0.2, callbacks=[early_stop])


def loss_y1_maker(task2_loss_multiplier):
    def inner(y_true, y_pred):
        y1_loss, y2_loss = loss_y1_y2(y_true, y_pred, task2_loss_multiplier)
        return y1_loss
    return inner


def loss_y2_maker(task2_loss_multiplier):
    def inner(y_true, y_pred):
        y1_loss, y2_loss = loss_y1_y2(y_true, y_pred, task2_loss_multiplier)
        return y2_loss
    return inner


def loss_y1_y2(y_true, y_pred, task2_loss_multiplier=1):
    # obtain masks
    mask_y1 = tf.math.is_nan(y_true[:, 1])
    mask_y2 = tf.math.logical_not(mask_y1)

    # fill values for differentiation
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)

    # split y values
    y1_true, y2_true = tf.split(y_true, [1, 1], 1)
    y1_pred, y2_pred = tf.split(y_pred, [1, 1], 1)

    # y1 loss
    y1_losses = tf.keras.metrics.binary_crossentropy(
        tf.expand_dims(y1_true[mask_y1], 1),
        tf.expand_dims(y1_pred[mask_y1], 1))
    y1_loss = tf.math.reduce_mean(y1_losses)

    # y2 loss
    y2_losses = tf.keras.metrics.mean_squared_error(y2_true[mask_y2], y2_pred[mask_y2])
    y2_loss = tf.math.reduce_mean(y2_losses) * task2_loss_multiplier

    return y1_loss, y2_loss
