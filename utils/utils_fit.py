"""Threshold search, loss and evaluation."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, f1_score, roc_curve, auc)


def _rates(y_true, y_score, thr):
    pred = (y_score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return tp, fp, tn, fn, tpr, tnr, (tp + tn) / max(len(y_true), 1)


def threshold_curve(y_true, y_score, grid=None):
    """Youden index J = TPR + TNR - 1 over a threshold grid."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    rows = []
    for t in grid:
        tp, fp, tn, fn, tpr, tnr, acc = _rates(y_true, y_score, t)
        rows.append(dict(thr=float(t), J=tpr + tnr - 1.0, acc=acc,
                         tpr=tpr, tnr=tnr, tp=tp, fp=fp, tn=tn, fn=fn))

    best = max(rows, key=lambda r: (r['J'], r['acc']))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pred = (y_score >= best['thr']).astype(int)

    best.update(auc=float(auc(fpr, tpr)), roc=(fpr, tpr),
                f1=float(f1_score(y_true, pred, zero_division=0)),
                grid=rows, y_true=y_true, y_score=y_score)
    return best


def balanced_threshold(y_true, y_score, grid=None):
    """Threshold maximising accuracy, ties broken by the Youden index."""
    res = threshold_curve(y_true, y_score, grid)
    top = max(r['acc'] for r in res['grid'])
    pool = [r for r in res['grid'] if r['acc'] >= top - 1e-12]
    best = max(pool, key=lambda r: (r['J'], -r['thr']))
    return best


def focal_loss(y_true, y_pred, alpha_pos=1.0, alpha_neg=1.0, gamma=2.0, s=1.0):
    """Class-weighted focal loss on a scalar sigmoid output."""
    y = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
    p = tf.clip_by_value(tf.reshape(y_pred, (-1,)), 1e-7, 1 - 1e-7)

    pos = tf.cast(y >= 0.5, tf.float32) * alpha_pos * y * \
        tf.pow(p, gamma) * -tf.math.log(p)
    neg = tf.cast(y <= 0.5, tf.float32) * alpha_neg * (1 - y) * \
        tf.pow(1 - p, gamma) * -tf.math.log(1 - p)
    return s * tf.reduce_mean(pos + neg)


def evaluate(model, dataset, thr=None):
    """Collect predictions and return threshold, rates, confusion matrix, AUC."""
    y_true, y_score = [], []
    for batch in dataset:
        y_score.extend(np.asarray(model(batch[0], training=False)).ravel().tolist())
        y_true.extend(np.asarray(batch[1]).ravel().tolist())

    res = threshold_curve(y_true, y_score)
    thr = res['thr'] if thr is None else thr
    res['thr_used'] = float(thr)
    res['confusion_matrix'] = confusion_matrix(
        np.asarray(y_true, int), (np.asarray(y_score) >= thr).astype(int))
    return res


def class_density(y_true, y_score, bins=50, sigma=2.0):
    """Prior-weighted score densities C0(s) and C1(s), plus their crossing.

    Each class histogram is smoothed with a Gaussian kernel and normalised by
    the total sample count, so the two curves are directly comparable.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    edges = np.linspace(0, 1, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n = len(y_score)

    def density(mask):
        h, _ = np.histogram(y_score[mask], bins=edges)
        r = max(int(3 * sigma), 1)
        k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
        h = np.convolve(h.astype(float), k / k.sum(), mode='same')
        return np.maximum(h, 0) / (n * width)

    c0, c1 = density(y_true <= 0.5), density(y_true >= 0.5)
    cross = np.where(np.diff(np.sign(c1 - c0)) != 0)[0]
    return dict(centres=centres, C0=c0, C1=c1,
                crossing=float(centres[cross[0]]) if len(cross) else 0.5)