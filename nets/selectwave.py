"""Ground motion selection under a multi-objective response criterion."""

import os
import numpy as np
import pandas as pd
from collections import Counter

from utils.utils import dual_reference_conformity

POS = ' '
NEG = ' '


def split_pools(paths, pos_pattern=POS, neg_pattern=NEG):
    """Split candidate paths into the two class pools."""
    pos = [p for p in paths if pos_pattern.lower() in str(p).lower()]
    neg = [p for p in paths if neg_pattern.lower() in str(p).lower()]
    return pos, neg


def vote_pool(model, dataset, pool, passes=30, top_n=2):
    """Monte-Carlo dropout voting over one candidate pool.

    Dropout stays active during inference, so every forward pass samples a
    different sub-network. The ``top_n`` highest-scoring candidates of each pass
    are recorded, and the candidates seen most often across all passes win.
    """
    wanted = set(pool)
    freq = Counter()

    for _ in range(passes):
        scores = {}
        for images, batch_paths in dataset:
            preds = np.asarray(model(images, training=True)).ravel()
            for path, score in zip(batch_paths, preds):
                key = path.decode() if isinstance(path, bytes) else str(path)
                if key in wanted:
                    scores[key] = float(score)
        for key, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]:
            freq[key] += 1

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
                 pos_pattern=POS, neg_pattern=NEG, passes=30, top_n=2):
    """Select the representative records of one group.

    The two pools are voted and averaged independently, so each class keeps its
    own target quantile.
    """
    pos_pool, neg_pool = split_pools(paths, pos_pattern, neg_pattern)
    if not pos_pool or not neg_pool:
        raise ValueError('empty candidate pool')

    pos_freq = vote_pool(model, dataset, pos_pool, passes, top_n)
    neg_freq = vote_pool(model, dataset, neg_pool, passes, top_n)
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


def select_all(model, dataset, paths, curve_of, target_of, prefix_of=None,
               passes=30, top_n=2, out_dir=None):
    """Apply :func:`select_group` to every group and record the DRC."""
    if prefix_of is None:
        def prefix_of(p):
            return os.path.splitext(os.path.basename(str(p)))[0].split('_')[0]

    groups = {}
    for p in paths:
        groups.setdefault(prefix_of(p), []).append(p)

    rows = []
    for prefix, subset in sorted(groups.items()):
        try:
            rec = select_group(model, dataset, subset, curve_of,
                               passes=passes, top_n=top_n)
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
