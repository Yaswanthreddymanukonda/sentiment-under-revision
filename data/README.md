# Data

Paired sentiment files from Kaushik et al. (CAD):

- https://github.com/acmi-lab/counterfactually-augmented-data
- commit `6f232a1d2a11462a30ce08fb4825b734ab30828e`
- folder `sentiment/combined/paired/`

`train.tsv` / `dev.tsv` / `test.tsv` are copies of the official `*_paired.tsv`
files. I didn't edit them. Rows go original, revision, original, revision, ...
Same `batch_id`, opposite labels.

`run_experiments.py` checks SHA-256 before training. HTML/quote cleanup is
in memory only. One train pair that also appears in dev is dropped at runtime
(see `data_audit.json`), not deleted from these files.

License from the original repo is in `LICENSE`.
