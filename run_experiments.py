# Train / eval for the sentiment revision experiments.
# Does adding the revised reviews actually help the model get both sides of a pair right?

import hashlib
import json
import platform
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'results'
OUT_DIR.mkdir(exist_ok=True)

# CAD paired files. This is the commit we pulled from; don't bump it without rechecking hashes.
DATA_URL = (
    'https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/'
    '6f232a1d2a11462a30ce08fb4825b734ab30828e/sentiment/combined/paired/'
)

SEED = 20260928
N_BOOT = 10000
C_GRID = [0.1, 1.0, 10.0]

# SHA256 of the tsvs in data/paired. If this fails, the files got swapped somehow.
FILE_HASHES = {
    'train': '7ac50ca3d70890f80e11ee59955cdbfc43f26b5078d0603717817ab224d2af18',
    'dev': '4c2d8a70cbce23eba27fb1f0bb5043cf044811912435cf8d10c3ea6fc7a51363',
    'test': '9534b20850205c59015b313983cae7afb31c54c04ffcbd19cea7f1bce5011275',
}

NEG_CUES = {'no', 'not', 'never', 'neither', 'nor', 'without'}

def save_json(name, obj):
    path = OUT_DIR / name
    path.write_text(
        json.dumps(obj, indent=2, default=lambda x: float(x)),
        encoding='utf-8',
    )


def clean_text(series):
    # IMDB leftover html + curly quotes. Don't write this back to disk.
    return (
        series.str.replace(r'<[^>]+>', ' ', regex=True)
        .str.replace('’', "'", regex=False)
    )


def load_data():
    """Load the three splits. Official files go original, revision, original, revision, ..."""
    data = {}
    for split in ['train', 'dev', 'test']:
        path = ROOT / 'data' / 'paired' / (split + '.tsv')
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            url = DATA_URL + split + '_paired.tsv'
            with urllib.request.urlopen(url, timeout=120) as resp:
                path.write_bytes(resp.read())

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != FILE_HASHES[split]:
            raise ValueError(
                f'Data checksum mismatch: {path}. Restore the pinned source file.'
            )

        df = pd.read_csv(path, sep='\t')
        assert not df.isna().any().any()

        orig = df.iloc[0::2].reset_index(drop=True)
        rev = df.iloc[1::2].reset_index(drop=True)

        # same batch_id, opposite labels. If this trips, the pairing is off.
        assert (orig.batch_id == rev.batch_id).all()
        assert orig.batch_id.is_unique
        assert (orig.Sentiment != rev.Sentiment).all()

        orig['Text'] = clean_text(orig.Text)
        rev['Text'] = clean_text(rev.Text)
        data[split] = {'orig': orig, 'new': rev}

    return data


def norm(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def add_negation_markers(text):
    """Stick NEG_ on the next few words after no/not/n't etc. Stops at punctuation.
    Not a parser, just a cheap local trick.
    """
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,;:]", text.lower())
    out = []
    scope = 0

    for tok in tokens:
        if re.fullmatch(r'[.!?,;:]', tok):
            scope = 0
            continue

        out.append(tok)
        if tok in NEG_CUES or tok.endswith("n't"):
            scope = 3
        elif scope:
            out.append('NEG_' + tok)
            scope -= 1

    return ' '.join(out)


def labels(df):
    return (df.Sentiment == 'Positive').to_numpy(dtype=int)


def make_vectorizer(feature):
    ngram = (1, 2) if feature == 'bigrams' else (1, 1)
    pre = add_negation_markers if feature == 'negation' else None
    return TfidfVectorizer(
        ngram_range=ngram,
        min_df=2,
        max_df=1.0,
        max_features=40000,
        sublinear_tf=True,
        preprocessor=pre,
        lowercase=True,
        strip_accents=None,
    )


def make_clf(alg, C):
    # liblinear was less fussy than lbfgs on these sparse matrices
    if alg == 'LR':
        return LogisticRegression(
            C=C, solver='liblinear', max_iter=3000, random_state=SEED
        )
    return LinearSVC(C=C, dual='auto', max_iter=10000, random_state=SEED)


def train_predict(train, dev, test, feature, alg):
    """Tune C on dev macro-F1, then score test. Don't peek at test while picking C."""
    vec = make_vectorizer(feature)
    X_train = vec.fit_transform(train.Text)
    X_dev = vec.transform(dev.Text)
    X_test = vec.transform(test.Text)

    y_train = labels(train)
    y_dev = labels(dev)

    grid = []
    best = None  # score, C, fitted model

    for C in C_GRID:
        model = make_clf(alg, C)
        model.fit(X_train, y_train)
        score = f1_score(y_dev, model.predict(X_dev), average='macro')
        grid.append({'C': C, 'dev_macro_f1': float(score)})
        if best is None or score > best[0]:
            best = (score, C, model)

    pred = best[2].predict(X_test)
    meta = {
        'C': best[1],
        'dev_macro_f1': best[0],
        'features': X_train.shape[1],
        'grid': grid,
    }
    return pred, meta


def evaluate(pred, y, n):
    """n is the number of pairs. First n rows are originals, last n are the revisions."""
    pred_o, pred_r = pred[:n], pred[n:]
    y_o, y_r = y[:n], y[n:]

    both_ok = (pred_o == y_o) & (pred_r == y_r)
    return {
        'orig_accuracy': accuracy_score(y_o, pred_o),
        'revised_accuracy': accuracy_score(y_r, pred_r),
        'orig_macro_f1': f1_score(y_o, pred_o, average='macro'),
        'revised_macro_f1': f1_score(y_r, pred_r, average='macro'),
        'joint_macro_f1': f1_score(y, pred, average='macro'),
        'pair_accuracy': np.mean(both_ok),
        'flip_rate': np.mean(pred_o != pred_r),
        'orig_confusion': confusion_matrix(y_o, pred_o).tolist(),
        'revised_confusion': confusion_matrix(y_r, pred_r).tolist(),
    }


def boot_interval(a, b=None):
    # resample pairs, not individual reviews, so orig/rev stay together
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), (N_BOOT, len(a)))
    if b is None:
        vals = a.astype(float)
    else:
        vals = a.astype(float) - b.astype(float)
    means = vals[idx].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()

def dump_audit(data):
    audit = {
        'files': {},
        'split_pairs': {},
        'cross_split_exact_overlap': {},
        'cross_split_pair_id_overlap': {},
        'removed_training_pairs': [],
    }

    for split in data:
        audit['split_pairs'][split] = len(data[split]['orig'])
        path = ROOT / 'data' / 'paired' / (split + '.tsv')
        rel = str(path.relative_to(ROOT))
        audit['files'][rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    # same cleaned text showing up in more than one split?
    text_sets = {}
    for split in data:
        texts = []
        for kind in data[split].values():
            texts.extend(norm(t) for t in kind.Text)
        text_sets[split] = set(texts)

    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        audit['cross_split_exact_overlap'][a + '_' + b] = len(
            text_sets[a] & text_sets[b]
        )
        overlap = len(
            set(data[a]['orig'].batch_id) & set(data[b]['orig'].batch_id)
        )
        audit['cross_split_pair_id_overlap'][a + '_' + b] = overlap
        assert overlap == 0

    holdout = text_sets['dev'] | text_sets['test']
    keep = []
    for o, r in zip(data['train']['orig'].Text, data['train']['new'].Text):
        keep.append(norm(o) not in holdout and norm(r) not in holdout)
    keep = np.array(keep)

    # if either side leaked into holdout, drop the whole pair
    audit['removed_training_pairs'] = np.where(~keep)[0].tolist()
    for kind in ['orig', 'new']:
        data['train'][kind] = data['train'][kind].loc[keep].reset_index(drop=True)

    # bail if cleaned dev/test text overlaps
    assert audit['cross_split_exact_overlap']['dev_test'] == 0
    audit['effective_training_pairs'] = int(keep.sum())
    return audit


def main():
    t0 = time.time()
    data = load_data()
    audit = dump_audit(data)

    dev = pd.concat([data['dev']['orig'], data['dev']['new']], ignore_index=True)
    test = pd.concat([data['test']['orig'], data['test']['new']], ignore_index=True)
    y = labels(test)
    n = len(data['test']['orig'])

    rows = []
    all_preds = {}

    for feature in ['unigrams', 'bigrams', 'negation']:
        for alg in ['LR', 'SVM']:
            for regime in ['original', 'augmented']:
                if regime == 'original':
                    train = data['train']['orig']
                else:
                    train = pd.concat(
                        [data['train']['orig'], data['train']['new']],
                        ignore_index=True,
                    )

                pred, meta = train_predict(train, dev, test, feature, alg)
                key = f'{alg}_{feature}_{regime}'
                all_preds[key] = pred

                row = {
                    'key': key,
                    'algorithm': alg,
                    'feature': feature,
                    'training': regime,
                    'training_examples': len(train),
                    **meta,
                    **evaluate(pred, y, n),
                }
                success = (pred[:n] == y[:n]) & (pred[n:] == y[n:])
                row['pair_accuracy_ci'] = boot_interval(success)
                rows.append(row)
                print(
                    key,
                    'orig', round(row['orig_accuracy'], 4),
                    'revised', round(row['revised_accuracy'], 4),
                    'pair', round(row['pair_accuracy'], 4),
                    flush=True,
                )

    # headline comparison — LR + bigrams
    base = all_preds['LR_bigrams_original']
    aug = all_preds['LR_bigrams_augmented']
    base_ok = (base[:n] == y[:n]) & (base[n:] == y[n:])
    aug_ok = (aug[:n] == y[:n]) & (aug[n:] == y[n:])
    improved = int(np.sum(aug_ok & ~base_ok))
    worsened = int(np.sum(base_ok & ~aug_ok))

    stats = {
        'primary': 'LR bigrams, augmented minus original',
        'pair_accuracy_difference': float(aug_ok.mean() - base_ok.mean()),
        'paired_bootstrap_95_ci': boot_interval(aug_ok, base_ok),
        'bootstrap_replicates': N_BOOT,
        'improved_pairs': improved,
        'worsened_pairs': worsened,
        'mcnemar_exact_two_sided_p': float(
            binomtest(improved, improved + worsened, 0.5).pvalue
        ),
    }

    # same # of training texts (1706). mixed = both sides of 853 pairs.
    budget = 2 * (len(data['train']['orig']) // 2)
    controls = []
    for seed in [11, 22, 33, 44, 55]:
        rng = np.random.default_rng(seed)
        orig_ix = rng.choice(len(data['train']['orig']), budget, replace=False)
        pair_ix = rng.choice(len(data['train']['orig']), budget // 2, replace=False)

        for regime in ['original', 'mixed']:
            if regime == 'original':
                train = data['train']['orig'].iloc[orig_ix]
            else:
                train = pd.concat(
                    [
                        data['train']['orig'].iloc[pair_ix],
                        data['train']['new'].iloc[pair_ix],
                    ],
                    ignore_index=True,
                )
            pred, meta = train_predict(train, dev, test, 'bigrams', 'LR')
            controls.append({
                'seed': seed,
                'training': regime,
                'training_examples': len(train),
                **meta,
                **evaluate(pred, y, n),
            })
            all_preds[f'control_{seed}_{regime}'] = pred

    # keyword hits only. not gold linguistic labels.
    orig_text = data['test']['orig'].Text.to_numpy()
    rev_text = data['test']['new'].Text.to_numpy()
    neg_re = re.compile(
        r"\b(?:no|not|never|neither|nor|without)\b|\b\w+n't\b", re.I
    )
    con_re = re.compile(
        r'\b(?:but|however|although|yet|nevertheless)\b', re.I
    )

    slices = []
    cue_masks = [
        (
            'negation cue',
            np.array([
                bool(neg_re.search(o) or neg_re.search(r))
                for o, r in zip(orig_text, rev_text)
            ]),
        ),
        (
            'contrast cue',
            np.array([
                bool(con_re.search(o) or con_re.search(r))
                for o, r in zip(orig_text, rev_text)
            ]),
        ),
    ]
    for label, mask in cue_masks:
        slices.append({
            'slice': label,
            'pairs': int(mask.sum()),
            'original_pair_accuracy': float(base_ok[mask].mean()),
            'augmented_pair_accuracy': float(aug_ok[mask].mean()),
        })

    pred_df = pd.DataFrame({
        'split': ['original'] * n + ['revised'] * n,
        'pair_id': list(range(n)) * 2,
        'source_batch_id': data['test']['orig'].batch_id.tolist() * 2,
        'true_label': y,
        **all_preds,
    })
    pred_df.to_csv(OUT_DIR / 'predictions.csv', index=False)

    # dump a few examples for the writeup
    cases = []
    for category, mask in [
        ('repaired', aug_ok & ~base_ok),
        ('regressed', base_ok & ~aug_ok),
        ('still_wrong', ~aug_ok & ~base_ok),
    ]:
        for i in np.where(mask)[0][:8]:
            cases.append({
                'category': category,
                'pair_id': int(i),
                'source_batch_id': int(data['test']['orig'].iloc[i].batch_id),
                'original_text': orig_text[i],
                'revised_text': rev_text[i],
                'original_label': int(y[i]),
                'revised_label': int(y[n + i]),
                'baseline_pred': [int(base[i]), int(base[n + i])],
                'augmented_pred': [int(aug[i]), int(aug[n + i])],
            })

    env = {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__,
        'seed': SEED,
        'elapsed_seconds': time.time() - t0,
        'source_repository': (
            'https://github.com/acmi-lab/counterfactually-augmented-data'
        ),
        'notes': (
            'sklearn only — no BERT / full IMDB run. '
            'pairs checked via batch_id; scoring is both sides of each pair.'
        ),
    }

    save_json('metrics.json', rows)
    save_json('primary_statistics.json', stats)
    save_json('matched_budget.json', controls)
    save_json('data_audit.json', audit)
    save_json('cue_slices.json', slices)
    save_json('error_cases.json', cases)
    save_json('environment.json', env)

    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
    pd.DataFrame(flat).to_csv(OUT_DIR / 'metrics.csv', index=False)

    print(json.dumps(stats, indent=2))
    print('Completed in', round(time.time() - t0, 1), 'seconds')


if __name__ == '__main__':
    main()
