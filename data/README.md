# Data placement

Configure local dataset paths in `configs/replication_config.json`.

NSL-KDD input may be the standard headerless `KDDTrain+` / `KDDTest+` form with 41 features plus label and optional difficulty column. UNSW-NB15 input must be CSV files containing `label`, with `attack_cat` available for multiclass experiments; `proto`, `service`, and `state` are treated as categorical when present.

The loaders never download data. Keep raw files under `data/raw/` or another local path ignored by Git.
