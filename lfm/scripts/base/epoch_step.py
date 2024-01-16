import tensorflow as tf


@tf.function
def train_step(users, items, rates, model, loss_object, optimizer, mae_metric):
    """
    Perform a single training step for a recommendation model.
    Args:
    - users: TensorFlow tensor representing user data.
    - items: TensorFlow tensor representing item data.
    - rates: TensorFlow tensor representing rating data.
    Returns:
    - total_loss: Total loss for the current training step.
    - mae: Mean Absolute Error (MAE) metric for the current batch.
    """
    with tf.device("/GPU:0"):
        with tf.GradientTape() as tape:
            users = tf.convert_to_tensor(users)
            items = tf.convert_to_tensor(items)
            rates = tf.convert_to_tensor(rates)
            inputs = (users, items)
            output_star = model(inputs, training=True)
            loss = loss_object(rates, output_star)
            total_loss = loss + tf.reduce_sum(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mae_metric.update_state(rates, output_star)
        mae = mae_metric.result()
    return total_loss, mae


@tf.function
def validation_step(users, items, rates, model, loss_object, mae_metric):
    """
    Perform a single validation step for a recommendation model.

    Args:
    - users: TensorFlow tensor representing user data.
    - items: TensorFlow tensor representing item data.
    - rates: TensorFlow tensor representing rating data.

    Returns:
    - val_loss: Validation loss for the current validation step.
    - val_mae: Mean Absolute Error (MAE) metric for the current batch.
    """
    users = tf.convert_to_tensor(users)
    items = tf.convert_to_tensor(items)
    rates = tf.convert_to_tensor(rates)
    inputs = (users, items)
    model.trainable = False
    output_star = model(inputs, training=False)
    val_loss = loss_object(rates, output_star)
    mae_metric.update_state(rates, output_star)
    val_mae = mae_metric.result()
    return val_loss, val_mae
