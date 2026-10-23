# CT-GMS

Multi-objective response-compatible ground motion selection.

Given a candidate suite already matched to a target spectrum, CT-GMS selects a
small subset whose nonlinear responses reproduce the multi-objective response
distribution of the full suite.

## Structure

```
CT-GMS/
├── S-pre.ipynb          RDTD generation, target quantiles, class labels
├── S-train.ipynb        classifier training
├── S-GMS.ipynb          Monte-Carlo dropout voting and record selection
├── nets/
│   ├── MobileNetV3_m.py     backbone with ECA and gated residual
│   └── selectwave.py        selection algorithm
└── utils/
    ├── dataloader.py        RDTD dataset
    ├── utils.py             quantiles, ARD / DRC
    ├── utils_aug.py         between-class augmentation
    ├── utils_fit.py         threshold search, loss, metrics
    └── callbacks.py         training controllers
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook          # run S-pre -> S-train -> S-GMS
```

Paths and parameters are declared in the first cell of each notebook. Data
directories are created on demand and excluded from version control.

## Requirements

Python 3.8+, TensorFlow 2.6+