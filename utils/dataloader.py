"""Dataset loading for response diagrams."""

import os
import pathlib

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def _decode(path, size=IMG_SIZE):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size)
    return tf.image.grayscale_to_rgb(img)


def _label_fn(class_names):
    """Map a path to the scalar class index, matching a sigmoid output."""
    index = {n: i for i, n in enumerate(class_names)}

    def _get(path):
        name = tf.strings.split(path, os.sep)[-2]

        def _scalar(b):
            return np.float32(index[b.numpy().decode()])
        label = tf.py_function(_scalar, [name], tf.float32)
        label.set_shape([])
        return label
    return _get


def list_images(root):
    """Return (paths, class_names) for a ``root/class/image`` layout."""
    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(str(p) for p in root.glob('*/*.[jp][pn]g'))
    if not paths:
        raise RuntimeError(f'no images under {root}')
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    return paths, names


def make_dataset(root, batch_size=32, train_ratio=0.8, size=IMG_SIZE,
                 seed=42, channels=3):
    """Build the training and validation splits.

    Parameters
    ----------
    train_ratio : float
        Share of images used for training; the remainder forms validation.
    channels : int
        ``3`` for plain diagrams, ``4`` to append a metadata channel.
    """
    paths, names = list_images(root)
    label = _label_fn(names)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(paths))
    n_train = int(len(paths) * train_ratio)
    splits = {
        True: [paths[i] for i in order[:n_train]],
        False: [paths[i] for i in order[n_train:]],
    }

    def _load(path):
        img = _decode(path, size)
        if channels == 4:
            pad = tf.zeros([size[0], size[1], 1], tf.float32)
            img = tf.concat([img, pad], axis=2)
        return img, label(path)

    out = {}
    for is_train, subset in splits.items():
        ds = tf.data.Dataset.from_tensor_slices(subset)
        if is_train:
            ds = ds.shuffle(len(subset), reshuffle_each_iteration=True, seed=seed)
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        out[is_train] = ds.batch(batch_size, drop_remainder=is_train).prefetch(
            tf.data.AUTOTUNE)

    return out[True], out[False], names


def make_predict_dataset(paths, batch_size=64, size=IMG_SIZE):
    """Batched inference loader returning (image, path)."""
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    ds = ds.map(lambda p: (_decode(p, size), p), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)