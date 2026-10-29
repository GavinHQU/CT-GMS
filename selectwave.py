"""Ground motion selection under a multi-objective response criterion."""

import os
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras.layers import BatchNormalization as _BatchNorm

from utils.utils import dual_reference_conformity


def _walk(model):
    """Yield every layer, including layers nested inside custom blocks.

    ``model.layers`` only exposes the top-level attributes of a subclassed
    model; the BatchNorm layers inside the custom ``Bottleneck`` blocks are
    reached through ``submodules`` (a plain ``Layer`` has no ``.layers``).
    """
    subs = getattr(model, 'submodules', None)
    if subs is not None:
        yield from subs
        return
    for layer in getattr(model, 'layers', []) or []:
        yield layer
        yield from _walk(layer)


def _set_mc_eval(model, on):
    """Pin every BatchNorm to moving statistics for Monte-Carlo inference.

    The ``BatchNorm`` subclass defined in ``MobileNetV3_m`` reads ``mc_eval`` and
    ignores the ``training`` flag, so dropout keeps sampling stochastic subnets
    while BN stays at its accumulated population statistics.
    """
    for layer in _walk(model):
        if isinstance(layer, _BatchNorm):
            layer.mc_eval = on


def _path_key(path):
    """Original path string of one dataset element (EagerTensor-safe)."""
    if hasattr(path, 'numpy'):
        path = path.numpy()
    if isinstance(path, bytes):
        return path.decode()
    return str(path)


def _pool_mean_scores(model, dataset, wanted, passes=1):
    """Mean model output per candidate over a few deterministic passes."""
    sums, counts = {}, {}
    for _ in range(passes):
        for images, batch_paths in dataset:
            preds = np.asarray(model(images, training=False)).ravel()
            for path, score in zip(batch_paths, preds):
                key = _path_key(path)
                if key in wanted:
                    sums[key] = sums.get(key, 0.0) + float(score)
                    counts[key] = counts.get(key, 0) + 1
    return {k: s / counts[k] for k, s in sums.items()}

POS = 'class_a'      # replace with your positive-class folder name
NEG = 'class_b'      # replace with your negative-class folder name


def split_pools(paths, pos_pattern=POS, neg_pattern=NEG):
    """Split candidate paths into the two class pools.

    Paths matching neither pattern are ignored.
    """
    pos = [p for p in paths if pos_pattern.lower() in str(p).lower()]
    neg = [p for p in paths if neg_pattern.lower() in str(p).lower()]
    return pos, neg


def folder_of(path):
    """Class-folder name of a candidate path."""
    return os.path.basename(os.path.dirname(str(path).rstrip('/\\')))


def _polarity(pool, label_map):
    """Selection direction for one pool, from its training label.

    ``label_map`` maps folder name -> class index (the alphabetical mapping
    used by ``utils.dataloader``). A pool whose class carries index 1 is
    maximised (polarity +1); index 0 is minimised (-1). Returns ``None`` when
    the mapping is unknown or the pool spans folders, in which case
    :func:`vote_pool` falls back to its median-score heuristic.
    """
    if not label_map or not pool:
        return None
    idxs = {label_map.get(folder_of(p)) for p in pool}
    idxs.discard(None)
    if len(idxs) == 1:
        return 1 if idxs.pop() == 1 else -1
    return None


def vote_pool(model, dataset, pool, passes=30, top_n=2, polarity=None):
    """Monte-Carlo dropout voting over one candidate pool.

    Dropout stays active during inference, so every forward pass samples a
    different sub-network. The ``top_n`` candidates ranked highest (or lowest,
    see ``polarity``) in each pass are recorded, and those seen most often
    across all passes win.

    polarity:
        +1 - pick the largest model outputs (the pool was trained with label 1).
        -1 - pick the smallest model outputs (the pool was trained with label 0).
        None - inferred from the pool's own median score. Pass an explicit
            ``label_map`` through :func:`select_group` when the classes are not
            well separated, so the choice follows the training label instead of
            the score distribution.
    """
    wanted = set(pool)
    freq = Counter()

    # BN uses moving statistics during MC inference; dropout stays stochastic.
    _set_mc_eval(model, True)
    try:
        if polarity is None:
            probe = _pool_mean_scores(model, dataset, wanted, passes=1)
            polarity = 1 if not probe or np.median(list(probe.values())) >= 0.5 else -1

        for _ in range(passes):
            scores = {}
            for images, batch_paths in dataset:
                preds = np.asarray(model(images, training=True)).ravel()
                for path, score in zip(batch_paths, preds):
                    key = _path_key(path)
                    if key in wanted:
                        scores[key] = float(score)
            ordered = sorted(scores.items(), key=lambda kv: kv[1],
                             reverse=(polarity > 0))
            for key, _ in ordered[:top_n]:
                freq[key] += 1
    finally:
        _set_mc_eval(model, False)

    return freq


def select_by_frequency(freq, top_n=2):
    """Candidates with the highest occurrence frequency."""
    return [p for p, _ in freq.most_common(top_n)]


def average_within_class(curves):
    """Mean of the curves selected within one class."""
    arrs = [np.asarray(c, float) for c in curves if c is not None and len(c)]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    return np.mean(np.vstack([a[:n] for a in arrs]), axis=0)


def select_group(model, dataset, paths, curve_of,
                 pos_pattern=POS, neg_pattern=NEG, label_map=None,
                 passes=30, top_n=2):
    """Select the representative records of one group.

    The two pools are voted and averaged independently, so each class keeps its
    own target quantile. With a ``label_map`` each pool is voted in the
    direction of its own training label; without one the vote direction is
    inferred from the scores, as before.
    """
    pos_pool, neg_pool = split_pools(paths, pos_pattern, neg_pattern)
    if not pos_pool or not neg_pool:
        raise ValueError('empty candidate pool')

    pos_freq = vote_pool(model, dataset, pos_pool, passes, top_n,
                         polarity=_polarity(pos_pool, label_map))
    neg_freq = vote_pool(model, dataset, neg_pool, passes, top_n,
                         polarity=_polarity(neg_pool, label_map))
    pos_paths = select_by_frequency(pos_freq, top_n)
    neg_paths = select_by_frequency(neg_freq, top_n)

    return {
        'pos': {'paths': pos_paths,
                'freq': [pos_freq[p] for p in pos_paths],
                'n_pool': len(pos_pool),
                'curve': average_within_class([curve_of(p) for p in pos_paths])},
        'neg': {'paths': neg_paths,
                'freq': [neg_freq[p] for p in neg_paths],
                'n_pool': len(neg_pool),
                'curve': average_within_class([curve_of(p) for p in neg_paths])},
    }


def select_all(model, dataset, paths, curve_of, target_of, label_map=None,
               prefix_of=None, passes=30, top_n=2, out_dir=None, ds_for=None):
    """Apply :func:`select_group` to every group and record the DRC.

    ``ds_for`` optionally builds a smaller inference dataset per group
    (``ds_for(subset)``), instead of scanning the full candidate set once per
    MC pass - a large speed-up when the pools are small.
    """
    if prefix_of is None:
        def prefix_of(p):
            # group id = text before the first underscore of the file name
            return os.path.splitext(os.path.basename(str(p)))[0].split('_')[0]

    groups = {}
    for p in paths:
        groups.setdefault(prefix_of(p), []).append(p)

    rows = []
    for prefix, subset in sorted(groups.items()):
        try:
            sub_ds = ds_for(subset) if ds_for is not None else dataset
            rec = select_group(model, sub_ds, subset, curve_of,
                               label_map=label_map, passes=passes, top_n=top_n)
        except ValueError:
            continue

        t_lo, t_hi = target_of(prefix)
        drc = None
        if t_lo is not None and rec['pos']['curve'] is not None:
            drc = dual_reference_conformity(rec['pos']['curve'], t_lo, t_hi)[0]

        rows.append({
            'group': prefix,
            'pos': ';'.join(os.path.basename(p) for p in rec['pos']['paths']),
            'pos_votes': rec['pos']['freq'],
            'neg': ';'.join(os.path.basename(p) for p in rec['neg']['paths']),
            'neg_votes': rec['neg']['freq'],
            'drc': drc,
        })

    table = pd.DataFrame(rows)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        table.to_csv(os.path.join(out_dir, 'selection.csv'), index=False)
    return table
