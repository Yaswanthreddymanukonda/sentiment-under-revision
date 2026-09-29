# Sentiment Under Revision

Train a sentiment model on original movie reviews vs originals + human revisions, then check if it labels **both** sides of each pair correctly.

Python 3.12, CPU only.

## Run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

Done when the last print is `"status": "PASS"`.

## Output

- `results/` — metrics, predictions, stats
- `figures/` — plots (PDF + PNG)

## Files

| File | What it does |
|---|---|
| `run_all.py` | runs everything |
| `run_experiments.py` | train / eval |
| `make_figures.py` | plots |
| `verify_results.py` | match against `reference_results/` |
| `data/paired/` | train / dev / test |

Data: [CAD sentiment pairs](https://github.com/acmi-lab/counterfactually-augmented-data) (Kaushik et al., ICLR 2020).
