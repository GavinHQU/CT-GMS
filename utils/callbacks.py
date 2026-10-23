"""Training callbacks: adaptive class weighting and bookkeeping."""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.utils_fit import balanced_threshold


class ThresholdController(tf.keras.callbacks.Callback):
    """Track the decision threshold and rebalance the two class weights.

    After each epoch the validation scores are thresholded at the accuracy
    optimum; the gap between the two class recalls then drives a small
    exponential update of the loss weights, and a sustained plateau stops
    training.
    """

    def __init__(self, val_data, alpha=(1.0, 1.0), bounds=(0.3, 3.0),
                 k=0.1, patience=3):
        super().__init__()
        self.val_data = val_data
        self.alpha_pos, self.alpha_neg = alpha
        self.lo, self.hi = bounds
        self.k = k
        self.patience = patience
        self._prev = None
        self._stall = 0
        self.threshold = 0.5

    def _scores(self):
        y, s = [], []
        for batch in self.val_data:
            s.extend(np.asarray(self.model(batch[0], training=False)).ravel().tolist())
            y.extend(np.asarray(batch[1]).ravel().tolist())
        return np.asarray(y), np.asarray(s)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_true, y_score = self._scores()
        if not len(y_true):
            return None

        res = balanced_threshold(y_true, y_score)
        self.threshold = res['thr']
        tpr, tnr = res['tpr'], res['tnr']
        gap = abs(tpr - tnr)

        if gap >= 0.4:                                   # one class clearly behind
            key = 'alpha_pos' if tpr < tnr else 'alpha_neg'
            val = getattr(self, key)
            setattr(self, key, float(np.clip(val + self.k * (self.hi - val) / self.lo,
                                             self.lo, self.hi)))
            mode = 'severe'
        elif gap > 0.03:                                 # drift, small steps
            behind_pos = self._prev is not None and (tpr - self._prev[0]) < (tnr - self._prev[1])
            key = 'alpha_pos' if behind_pos else 'alpha_neg'
            val = getattr(self, key)
            setattr(self, key, float(np.clip(val + self.k * (self.hi - val), self.lo, self.hi)))
            mode = 'mild'
        else:
            mode = 'balanced'
            still = self._prev is not None and \
                (tpr - self._prev[0]) < 2e-3 and (tnr - self._prev[1]) < 2e-3
            self._stall = self._stall + 1 if still else 0
            if self._stall >= self.patience:
                self.stop_training = True

        self._prev = (tpr, tnr)
        logs.update(thr=self.threshold, tpr=tpr, tnr=tnr, gap=gap,
                    auc=res['auc'], f1=res['f1'], mode=mode,
                    alpha_pos=self.alpha_pos, alpha_neg=self.alpha_neg)
        return res

    def weights(self):
        return self.alpha_pos, self.alpha_neg


class History(tf.keras.callbacks.Callback):
    """Append per-epoch metrics to a CSV."""

    def __init__(self, out_dir, name='history.csv'):
        super().__init__()
        self.path = os.path.join(out_dir, name)
        self.rows = []
        os.makedirs(out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        row = {'epoch': epoch + 1}
        row.update({k: float(v) for k, v in (logs or {}).items()
                    if isinstance(v, (int, float, np.floating))})
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


class SaveWeights(tf.keras.callbacks.Callback):
    """Save weights periodically and whenever the monitored loss improves."""

    def __init__(self, out_dir, period=10, monitor='val_loss'):
        super().__init__()
        self.dir = out_dir
        self.period = max(int(period), 1)
        self.monitor = monitor
        self.best = np.inf
        os.makedirs(out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is None:
            return
        if (epoch + 1) % self.period == 0:
            self.model.save_weights(os.path.join(self.dir, f'ep{epoch+1:03d}.weights.h5'))
        if float(v) < self.best:
            self.best = float(v)
            self.model.save_weights(os.path.join(self.dir, 'best.weights.h5'))


def cosine_schedule(init_lr=1e-4, min_lr=1e-6, total_steps=1e4, warmup=0):
    """Cosine decay with optional linear warmup.

    ``total_steps`` is the number of optimisation steps, i.e.
    ``steps_per_epoch * epochs``.
    """
    class _Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            total, w = float(total_steps), float(max(warmup, 1))
            warm_lr = init_lr * step / w
            cos_lr = min_lr + 0.5 * (init_lr - min_lr) * (
                1 + tf.cos(np.pi * tf.maximum(step - w, 0.0) / max(total - w, 1.0)))
            return tf.where(step < w, warm_lr, cos_lr)
    return _Schedule()