"""Threshold search and loss."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_curve, auc


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
    """Class-weighted focal loss on a scalar sigmoid output.

    The two terms weight the positive and negative focal losses by the target
    itself: a mixed sample whose label is ``t`` receives ``t`` of the positive
    term and ``1 - t`` of the negative one, so the loss of a between-class
    mixture is exactly the interpolated loss of its two source diagrams, as
    documented. Crisp 0/1 labels reduce to the standard weighted binary focal
    loss, numerically identical to the previous masked formulation.

    (The earlier masked version dropped the complementary term for soft
    labels, so a 0.3/0.7 mixture was punished with the wrong-side loss.)
    """
    y = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
    p = tf.clip_by_value(tf.reshape(y_pred, (-1,)), 1e-7, 1 - 1e-7)

    pos = alpha_pos * y * tf.pow(1 - p, gamma) * -tf.math.log(p)
    neg = alpha_neg * (1 - y) * tf.pow(p, gamma) * -tf.math.log(1 - p)
    return s * tf.reduce_mean(pos + neg)
