"""Between-class augmentation for response diagrams."""

import tensorflow as tf


def mix(x, y, paths=None, mode='interior', training=True):
    """Linear interpolation between shuffled pairs of a batch.

    ``mode='interior'`` keeps the weight centred; ``mode='boundary'`` pushes it
    towards either end, yielding samples close to a single class. When ``paths``
    is given the returned tag records which two diagrams produced each sample.
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    b = tf.shape(x)[0]

    if not training:
        return x, y, tf.zeros((b, 1), tf.float32), paths

    shape = (b, 1, 1, 1)
    if mode == 'boundary':
        lam = tf.where(tf.random.uniform(shape) < 0.5,
                       tf.random.uniform(shape, 0.02, 0.24),
                       tf.random.uniform(shape, 0.76, 0.98))
    else:
        lam = tf.random.uniform(shape, 0.25, 0.75)
    flat = tf.reshape(lam, (b, 1))

    order = tf.argsort(tf.random.uniform([b]))
    x2, y2 = tf.gather(x, order), tf.gather(y, order)

    tag = None
    if paths is not None:
        p2 = tf.gather(paths, order)
        tag = tf.strings.reduce_join(
            tf.stack([paths, p2], axis=1), axis=1, separator=',')

    return lam * x + (1.0 - lam) * x2, flat * y + (1.0 - flat) * y2, flat, tag