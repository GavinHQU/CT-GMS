"""MobileNetV3-m: lightweight classifier for elastic time-domain response diagrams."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class BatchNorm(layers.BatchNormalization):
    """BatchNorm pinned to moving statistics during Monte-Carlo inference.

    When ``mc_eval`` is set, ``call`` ignores the ``training`` flag and always
    uses the accumulated population statistics, while dropout layers elsewhere
    stay stochastic. This removes the batch-statistic variance shift that would
    otherwise corrupt MC-dropout uncertainty estimates.
    """

    def call(self, x, training=False):
        if getattr(self, 'mc_eval', False):
            training = False
        return super().call(x, training=training)


class ECA(layers.Layer):
    """Efficient channel attention via 1-D convolution across channels."""

    def __init__(self, channels, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        t = int(abs((np.log2(channels) + b) / gamma))
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv = layers.Conv1D(1, t if t % 2 else t + 1,
                                  padding='same', use_bias=False,
                                  activation='sigmoid')

    def call(self, x):
        y = tf.transpose(self.gap(x), (0, 3, 1, 2))
        y = self.conv(y)
        return x * tf.transpose(y, (0, 2, 3, 1))


class GatedAdd(layers.Layer):
    """y = x + sigmoid(g) * Dropout(r); L2 penalty on the gate g."""

    def __init__(self, channels, drop=0.2, l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.drop = layers.Dropout(drop)
        self.l2 = l2
        self.gate = self.add_weight(name='gate', shape=(1, 1, 1, channels),
                                    initializer='ones', trainable=True)

    def call(self, inputs, training=False):
        x, r = inputs
        return x + tf.sigmoid(self.gate) * self.drop(r, training=training)

    def reg_loss(self):
        return 0.5 * self.l2 * tf.reduce_sum(tf.square(self.gate))


class Bottleneck(layers.Layer):
    """Inverted residual block with linear bottleneck."""

    def __init__(self, cin, cout, stride=1, k=3, exp=4, eca=True,
                 drop=0.2, l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        hidden = cin * exp
        self.expand = layers.Conv2D(hidden, 1, use_bias=False)
        self.bn1 = BatchNorm()
        self.dw = layers.DepthwiseConv2D(k, stride, padding='same', use_bias=False)
        self.bn2 = BatchNorm()
        self.proj = layers.Conv2D(cout, 1, use_bias=False)
        self.bn3 = BatchNorm()
        self.eca = ECA(cout) if eca else None
        self.skip = (stride == 1 and cin == cout)
        self.add = GatedAdd(cout, drop, l2)

    def call(self, x, training=False):
        r = tf.nn.relu6(self.bn1(self.expand(x), training=training))
        r = tf.nn.relu6(self.bn2(self.dw(r), training=training))
        r = self.bn3(self.proj(r), training=training)
        if self.eca is not None:
            r = self.eca(r)
        if self.skip:
            return self.add([x, r], training=training)
        return tf.nn.relu6(r)


class MobileNetV3_m(tf.keras.Model):
    """Classifier head: 224x224xC input, scalar sigmoid or softmax output.

    Parameters
    ----------
    input_shape : tuple
        ``(224, 224, C)``; ``C`` is the channel count produced by the
        dataloader (``3`` for plain diagrams, the default on both sides).
    num_classes : int
        ``1`` for a sigmoid probability, otherwise softmax over classes.
    mid_units, dropout_rate : int, float
        Width of the penultimate layer and its dropout.
    """

    CFG = [
        (16, 3, 1, 1, False), (24, 3, 2, 3, False), (24, 3, 1, 3, False),
        (40, 3, 2, 3, True), (40, 3, 1, 3, True), (40, 3, 1, 3, True),
        (80, 3, 2, 3, False), (80, 3, 1, 3, False), (80, 3, 1, 3, False),
        (80, 3, 1, 3, False), (112, 3, 1, 3, True), (112, 3, 1, 3, True),
        (160, 3, 2, 4, True), (160, 3, 1, 4, True), (960, 3, 1, 4, True),
    ]

    def __init__(self, input_shape=(224, 224, 3), num_classes=1,
                 mid_units=128, dropout_rate=0.2, l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.stem = layers.Conv2D(16, 3, 2, padding='same', use_bias=False)
        self.stem_bn = BatchNorm()

        blocks, cin = [], 16
        for i, (cout, k, s, exp, eca) in enumerate(self.CFG):
            blocks.append(Bottleneck(cin, cout, s, k, exp, eca,
                                     drop=dropout_rate, l2=l2,
                                     name=f'bneck{i:02d}'))
            cin = cout
        self.blocks = blocks

        self.head = layers.Conv2D(128, 1, use_bias=False)
        self.head_bn = BatchNorm()
        self.gap = layers.GlobalAveragePooling2D()
        self.drop = layers.Dropout(dropout_rate)
        self.fc1 = layers.Dense(mid_units, activation='relu')
        act = 'sigmoid' if num_classes == 1 else 'softmax'
        self.fc2 = layers.Dense(num_classes, activation=act)

    def call(self, x, training=False):
        x = tf.nn.relu6(self.stem_bn(self.stem(x), training=training))
        for blk in self.blocks:
            x = blk(x, training=training)
        x = tf.nn.relu6(self.head_bn(self.head(x), training=training))
        x = self.drop(self.gap(x), training=training)
        return self.fc2(self.fc1(x))

    def reg_loss(self):
        return tf.add_n([b.add.reg_loss() for b in self.blocks if b.skip])


def build_classifier(input_channels=3, num_classes=1, mid_units=128,
                     dropout_rate=0.2):
    """Construct and build MobileNetV3-m."""
    model = MobileNetV3_m(input_shape=(224, 224, input_channels),
                          num_classes=num_classes,
                          mid_units=mid_units,
                          dropout_rate=dropout_rate)
    model.build((None, 224, 224, input_channels))
    return model
