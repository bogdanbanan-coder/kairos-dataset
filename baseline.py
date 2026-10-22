"""Compare feature configurations and classifiers on the held-out authors.

The text configurations add one representation of the post to the baseline
numerical and categorical features. Media is not used.

    python baseline.py
    python baseline.py --data D:/kairos

Writes one metrics CSV per classifier and one prediction CSV per model/configuration
pair, in the format evaluate.py reads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Gradient boosting asks joblib for the number of physical cores, joblib asks
# `wmic`, which Windows 11 no longer ships, and a warning and a traceback follow.
# joblib skips the question only for a cap below the logical count; half of it
# is the physical count on most machines.
if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from scipy import sparse  # noqa: E402

from kairos import features, text_features  # noqa: E402
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts  # noqa: E402
from kairos.label import TYPICAL, author_level  # noqa: E402
from kairos.metrics import score  # noqa: E402

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
NUMERIC = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
CONFIGURATIONS = {
    "baseline": (NUMERIC, None),
    "estimates": (NUMERIC + features.ESTIMATES_NUM, None),
    "text_tf_idf": (NUMERIC, "text_tf_idf"),
    "text_tf_idf_truncatedsvd": (NUMERIC, "text_tf_idf_truncatedsvd"),
    "text_multilingual_e5_small": (NUMERIC, "text_multilingual_e5_small"),
}


def preprocess(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                features.CONTENT_CAT,
            ),
        ],
        sparse_threshold=0.0,
    )


def gradient_boosting() -> HistGradientBoostingClassifier:
    # Balanced weights counter the 20 / 60 / 20 class mix for macro-F1.
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=SEED,
    )


def logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000
    )


MODELS = {
    "hist_gradient_boosting": gradient_boosting,
    "logistic_regression": logistic_regression,
}


def add_text(base, text, *, dense: bool):
    """Append one text representation; keep TF-IDF sparse for regression."""
    if text is None:
        return base
    base = base.astype(np.float32)
    if sparse.issparse(text):
        if dense:
            text = text.toarray()
        else:
            return sparse.hstack((base, text), format="csr")
    return np.hstack((base, text))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and compare classifiers.")
    ap.add_argument(
        "--data", default=DATA_DIR, help="unpacked release (default: data/)"
    )
    args = ap.parse_args()
    t0 = time.time()

    labels = load_labels(args.data)
    levelled = author_level(load_posts(args.data))
    table = labels.merge(
        features.build(levelled, load_followers(args.data)),
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )
    if table["level"].isna().any():
        raise SystemExit("some labelled posts have no level: is the release complete?")

    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    print(
        f"train {len(train):,} posts / {train['author'].nunique():,} authors   "
        f"test {len(test):,} posts / {test['author'].nunique():,} authors",
        flush=True,
    )

    # Two answers that need no model, for scale.
    mix = np.bincount(train["label"], minlength=3) / len(train)
    reference = {
        "always_typical": score(test["label"], np.full(len(test), TYPICAL)),
        "random_in_train_mix": score(
            test["label"], np.random.default_rng(SEED).choice(3, size=len(test), p=mix)
        ),
    }

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    results_by_model = {name: {} for name in MODELS}
    prepared = {}
    text_representations = None
    for name, (numeric_features, text_name) in CONFIGURATIONS.items():
        key = tuple(numeric_features)
        if key not in prepared:
            transformer = preprocess(numeric_features)
            prepared[key] = (
                transformer.fit_transform(train),
                transformer.transform(test),
            )
        base_train, base_test = prepared[key]
        if text_name is not None and text_representations is None:
            text_representations = text_features.build(
                train["text"].fillna("").tolist(),
                test["text"].fillna("").tolist(),
                RESULTS,
            )
        text_train, text_test = (
            text_representations[text_name] if text_name is not None else (None, None)
        )
        for model_name, make_model in MODELS.items():
            dense = model_name == "hist_gradient_boosting"
            x_train = add_text(base_train, text_train, dense=dense)
            x_test = add_text(base_test, text_test, dense=dense)
            predicted = make_model().fit(x_train, train["label"]).predict(x_test)
            results_by_model[model_name][name] = score(test["label"], predicted)
            test[["author", "post"]].assign(label=predicted).to_csv(
                RESULTS / "predictions" / f"{model_name}__{name}.csv", index=False
            )
            print(f"{model_name} / {name}: done", flush=True)

    for model_name, results in results_by_model.items():
        results.update(reference)
        pd.DataFrame(
            [{"name": name, **metrics} for name, metrics in results.items()],
            columns=[
                "name",
                "macro_f1",
                "f1_below",
                "f1_typical",
                "f1_above",
                "qwk",
                "accuracy",
            ],
        ).to_csv(RESULTS / "metrics" / f"{model_name}.csv", index=False)

        name_width = max(20, *(len(name) for name in results))
        print(f"\n{model_name}")
        print(f"{'':{name_width}s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
        for name, m in results.items():
            print(
                f"{name:{name_width}s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}"
            )

    print(
        f"\nwrote results/metrics/*.csv and results/predictions/*.csv "
        f"({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
