"""Feature columns, their computation, and train-fitted preprocessing.

content    what the post is: length, hashtags, mentions, links, type, posting time
history    who wrote it: the author's level and spread, rhythm, follower count
artefact   how it was collected: the post's age at crawl and its place in the
           author's run
text       TF-IDF, normalized SVD, and multilingual E5 vectors
comparison text similarity to the author's earlier posts and their residuals
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    Normalizer,
    OneHotEncoder,
    StandardScaler,
)

from kairos.label import MIN_HISTORY

# The release replaces links, handles and names with placeholders such as ⟨URL⟩,
# ⟨HANDLE⟩ and ⟨PERSON_3⟩, so the counters look for those as well as raw text.
_URL = re.compile(r"https?://|⟨URL⟩")
_MENTION = re.compile(r"@\w+|⟨HANDLE⟩|⟨(?:PERSON|AUTHOR)[^⟩]*⟩")
_HASHTAG = re.compile(r"#(?:\w+|⟨[^⟩]*⟩)")

CONTENT_NUM = [
    "n_chars",
    "n_words",
    "n_lines",
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "has_question",
    "is_textless",
    "hour",
    "weekday",
]
CONTENT_CAT = ["content_type"]
HISTORY_NUM = [
    "level",
    "hist_iqr",
    "days_since_prev",
    "log_followers",
]
ARTEFACT_NUM = ["age_days", "hist_n"]

# history
ESTIMATES_NUM = ["hist_mean", "hist_sample_var"]

N_LAST = 20
N_NEAREST = 5
TEXT_COMPARISON_E5_NUM = [
    "similarity_all_e5",
    "similarity_last_5_e5",
    "last_nearest_resid_e5",
]
TEXT_COMPARISON_TFIDF_NUM = [
    "similarity_all_tfidf",
    "similarity_last_5_tfidf",
    "last_nearest_resid_tfidf",
]

TFIDF_MAX_FEATURES = 5000
SVD_COMPONENTS = 128
E5_DIMENSIONS = 384
E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

# TF-IDF and SVD use only the leading names matching their actual dimensions.
TEXT_TFIDF_NUM = [f"text_tfidf_{i:04d}" for i in range(TFIDF_MAX_FEATURES)]
TEXT_SVD_NUM = [f"text_svd_{i:03d}" for i in range(SVD_COMPONENTS)]
TEXT_E5_NUM = [f"text_e5_{i:03d}" for i in range(E5_DIMENSIONS)]
TEXT_NUM = TEXT_TFIDF_NUM + TEXT_SVD_NUM + TEXT_E5_NUM


def build(levelled: pd.DataFrame, followers: pd.Series) -> pd.DataFrame:
    """Every feature for every own post.

    Takes the output of `label.author_level`, sorted by author and time, and
    reads each author's history from earlier posts only, as the level does.
    """
    df = pd.DataFrame(index=levelled.index)
    df["author"] = levelled["author"]
    df["post"] = levelled["post"]
    df["content_type"] = levelled["content_type"]
    df["level"] = levelled["level"]
    df["hist_n"] = levelled["hist_n"]
    text = levelled["text"]
    df["text"] = text
    df["n_chars"] = text.str.len()
    df["n_words"] = text.str.split().str.len().fillna(0)
    df["n_lines"] = text.str.count("\n") + 1
    df["n_hashtags"] = text.str.count(_HASHTAG)
    df["n_mentions"] = text.str.count(_MENTION)
    df["n_urls"] = text.str.count(_URL)
    df["has_question"] = text.str.contains("?", regex=False).astype(int)
    df["is_textless"] = (df["n_chars"] == 0).astype(int)
    df["hour"] = levelled["posted_at"].dt.hour
    df["weekday"] = levelled["posted_at"].dt.weekday

    # Reactions are one snapshot taken at crawl time, and an older post has had
    # longer to collect them. Age is a property of the collection, not of the
    # post, so it sits in its own group instead of hiding among the content.
    df["age_days"] = (
        levelled["crawled_at"] - levelled["posted_at"]
    ).dt.total_seconds() / 86400
    df["age_days"] = df["age_days"].fillna(df["age_days"].median())

    df["log_followers"] = np.log1p(
        levelled["author"].map(followers).fillna(0).clip(lower=0)
    )

    df["hist_iqr"] = 0.0
    df["hist_mean"] = 0.0
    df["hist_sample_var"] = 0.0
    df["days_since_prev"] = -1.0

    # `levelled` is sorted by author and publication time. For each post, use
    # only the history collected before adding that post to it.
    for _, posts in levelled.groupby("author", sort=False):
        history = []
        previous_posted_at = None

        for author_post in posts.itertuples():
            if len(history) >= MIN_HISTORY:
                q25, q75 = np.quantile(history, [0.25, 0.75])
                df.at[author_post.Index, "hist_iqr"] = q75 - q25

                n = len(history)
                mean = 1 / n * sum(history)
                df.at[author_post.Index, "hist_mean"] = mean
                df.at[author_post.Index, "hist_sample_var"] = (
                    1 / (n - 1) * sum((mean - el) ** 2 for el in history)
                )

            if (
                previous_posted_at is not None
                and pd.notna(previous_posted_at)
                and pd.notna(author_post.posted_at)
            ):
                df.at[author_post.Index, "days_since_prev"] = (
                    author_post.posted_at - previous_posted_at
                ).total_seconds() / 86400

            history.append(author_post.log_reactions)
            previous_posted_at = author_post.posted_at

    return df


def _e5_embeddings(texts: list[str], results_dir: Path) -> np.ndarray:
    """Encode each whole post up to 512 tokens; reuse the result on later runs."""
    digest = hashlib.sha256(f"{E5_MODEL}@{E5_REVISION}:512:query".encode())
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    cache_dir = results_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"multilingual_e5_small_{digest.hexdigest()[:16]}.npy"
    if cache_file.is_file():
        return np.load(cache_file, mmap_mode="r")

    from sentence_transformers import SentenceTransformer

    model_cache = results_dir / "model_cache"
    snapshot = (
        model_cache
        / f"models--{E5_MODEL.replace('/', '--')}"
        / "snapshots"
        / E5_REVISION
    )
    model = SentenceTransformer(
        E5_MODEL,
        revision=E5_REVISION,
        cache_folder=str(model_cache),
        local_files_only=all(
            (snapshot / name).is_file()
            for name in (
                "modules.json",
                "config.json",
                "tokenizer.json",
                "model.safetensors",
            )
        ),
    )
    model.max_seq_length = 512
    indices = [i for i, text in enumerate(texts) if text.strip()]
    embeddings = np.zeros(
        (len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32
    )
    if indices:
        embeddings[indices] = model.encode(
            ["query: " + texts[i] for i in indices],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    temporary = cache_file.with_suffix(".tmp.npy")
    np.save(temporary, embeddings)
    temporary.replace(cache_file)
    return embeddings


def _text_comparison(history: pd.DataFrame, vectors) -> np.ndarray:
    """Compare each post with strictly earlier posts of the same author.

    Rows of vectors follow history's row order. Missing dates cannot establish
    an earlier post. Missing residuals are excluded only from nearest neighbours.
    """
    if (
        type(N_LAST) is not int
        or type(N_NEAREST) is not int
        or not 1 <= N_NEAREST <= N_LAST
    ):
        raise ValueError(
            "N_LAST and N_NEAREST must be integers: 1 <= N_NEAREST <= N_LAST"
        )

    result = np.zeros((len(history), 3), dtype=np.float32)
    history = history.reset_index(drop=True)
    for _, posts in history.groupby("author", sort=False):
        posts = posts.dropna(subset=["posted_at"]).sort_values(
            "posted_at", kind="mergesort"
        )
        positions = posts.index.to_numpy()
        author_vectors = vectors[positions]
        residuals = posts["resid"].to_numpy(dtype=np.float64, na_value=np.nan)
        history_sum = np.zeros(vectors.shape[1], dtype=np.float64)
        start = 0
        # Process equal timestamps together so they cannot enter each other's history.
        for _, simultaneous in posts.groupby("posted_at", sort=False):
            end = start + len(simultaneous)
            current = author_vectors[start:end]
            if start:
                last_5_mean = np.asarray(
                    author_vectors[max(0, start - 5) : start].mean(axis=0)
                ).reshape(-1)
                means = np.stack((history_sum / start, last_5_mean))
                result[positions[start:end], :2] = cosine_similarity(current, means)

                last_start = max(0, start - N_LAST)
                candidates = (
                    np.flatnonzero(np.isfinite(residuals[last_start:start])) + last_start
                )[::-1]
                if len(candidates):
                    similarities = cosine_similarity(current, author_vectors[candidates])
                    # Candidates are newest first; stable sorting resolves equal cosines.
                    nearest = np.argsort(-similarities, axis=1, kind="stable")[
                        :, :N_NEAREST
                    ]
                    weighted = similarities * residuals[candidates]
                    result[positions[start:end], 2] = np.take_along_axis(
                        weighted, nearest, axis=1
                    ).sum(axis=1)

            history_sum += np.asarray(current.sum(axis=0)).reshape(-1)
            start = end

    return result


def prepare(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
    *,
    history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add requested text vectors and history comparisons, fitting on train only.

    Comparisons require the full output of author_level(), including the first
    unlabelled posts. Only earlier posts contribute to each post's comparisons.
    """
    requested = set(feature_names)
    compare_tfidf = bool(requested.intersection(TEXT_COMPARISON_TFIDF_NUM))
    compare_e5 = bool(requested.intersection(TEXT_COMPARISON_E5_NUM))
    need_comparison = compare_tfidf or compare_e5
    need_tfidf = bool(requested.intersection(TEXT_TFIDF_NUM)) or compare_tfidf
    need_svd = bool(requested.intersection(TEXT_SVD_NUM))
    need_e5 = bool(requested.intersection(TEXT_E5_NUM)) or compare_e5
    if not (need_tfidf or need_svd or need_e5):
        return train, test

    train_text = train["text"].fillna("").tolist()
    test_text = test["text"].fillna("").tolist()
    texts = train_text + test_text
    train_rows, test_rows = slice(0, len(train)), slice(len(train), None)
    if need_comparison:
        if history is None:
            raise ValueError(
                "text comparison features require history=author_level(posts)"
            )
        authors = pd.concat([train["author"], test["author"]]).unique()
        history = history.loc[
            history["author"].isin(authors),
            ["author", "post", "posted_at", "text", "resid"],
        ].reset_index(drop=True)
        post_index = pd.MultiIndex.from_frame(history[["author", "post"]])
        if not post_index.is_unique:
            raise ValueError("history must have unique (author, post) keys")
        train_rows = post_index.get_indexer(
            pd.MultiIndex.from_frame(train[["author", "post"]])
        )
        test_rows = post_index.get_indexer(
            pd.MultiIndex.from_frame(test[["author", "post"]])
        )
        if (train_rows < 0).any() or (test_rows < 0).any():
            raise ValueError("some train/test posts are missing from history")
        texts = history["text"].fillna("").tolist()
    train_parts, test_parts = [train], [test]

    if need_tfidf or need_svd:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=2,
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        vectorizer.fit(train_text)
        tfidf = vectorizer.transform(texts)
        tfidf_train, tfidf_test = tfidf[train_rows], tfidf[test_rows]
        if requested.intersection(TEXT_TFIDF_NUM):
            columns = TEXT_TFIDF_NUM[: tfidf_train.shape[1]]
            train_parts.append(
                pd.DataFrame.sparse.from_spmatrix(
                    tfidf_train, index=train.index, columns=columns
                )
            )
            test_parts.append(
                pd.DataFrame.sparse.from_spmatrix(
                    tfidf_test, index=test.index, columns=columns
                )
            )
        if need_svd:
            lsa = make_pipeline(
                TruncatedSVD(
                    n_components=min(SVD_COMPONENTS, *tfidf_train.shape),
                    random_state=42,
                ),
                Normalizer(),
            )
            svd_train = lsa.fit_transform(tfidf_train).astype(np.float32)
            svd_test = lsa.transform(tfidf_test).astype(np.float32)
            columns = TEXT_SVD_NUM[: svd_train.shape[1]]
            train_parts.append(
                pd.DataFrame(svd_train, index=train.index, columns=columns)
            )
            test_parts.append(pd.DataFrame(svd_test, index=test.index, columns=columns))
        if compare_tfidf:
            comparison = _text_comparison(history, tfidf)
            train_parts.append(
                pd.DataFrame(
                    comparison[train_rows],
                    index=train.index,
                    columns=TEXT_COMPARISON_TFIDF_NUM,
                )
            )
            test_parts.append(
                pd.DataFrame(
                    comparison[test_rows],
                    index=test.index,
                    columns=TEXT_COMPARISON_TFIDF_NUM,
                )
            )

    if need_e5:
        e5 = _e5_embeddings(texts, results_dir)
        if requested.intersection(TEXT_E5_NUM):
            train_parts.append(
                pd.DataFrame(e5[train_rows], index=train.index, columns=TEXT_E5_NUM)
            )
            test_parts.append(
                pd.DataFrame(e5[test_rows], index=test.index, columns=TEXT_E5_NUM)
            )
        if compare_e5:
            comparison = _text_comparison(history, e5)
            train_parts.append(
                pd.DataFrame(
                    comparison[train_rows],
                    index=train.index,
                    columns=TEXT_COMPARISON_E5_NUM,
                )
            )
            test_parts.append(
                pd.DataFrame(
                    comparison[test_rows],
                    index=test.index,
                    columns=TEXT_COMPARISON_E5_NUM,
                )
            )

    return pd.concat(train_parts, axis=1), pd.concat(test_parts, axis=1)


def _tfidf_matrix(frame: pd.DataFrame):
    return frame.sparse.to_coo().tocsr()


def preprocess(feature_names: list[str]) -> ColumnTransformer:
    """Scale ordinary numbers, encode categories, and pass text vectors through."""
    categorical_names = set(CONTENT_CAT)
    tfidf_names = set(TEXT_TFIDF_NUM)
    vector_names = set(TEXT_SVD_NUM + TEXT_E5_NUM)
    unscaled_names = categorical_names | set(TEXT_NUM)
    categorical = [name for name in feature_names if name in categorical_names]
    tfidf = [name for name in feature_names if name in tfidf_names]
    vectors = [name for name in feature_names if name in vector_names]
    numeric = [name for name in feature_names if name not in unscaled_names]
    scaler = StandardScaler()
    if tfidf or vectors:
        scaler = make_pipeline(
            scaler, FunctionTransformer(np.asarray, kw_args={"dtype": np.float32})
        )
    transformers = []
    if numeric:
        transformers.append(("num", scaler, numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                    dtype=np.float32 if tfidf or vectors else np.float64,
                ),
                categorical,
            )
        )
    if tfidf:
        transformers.append(
            (
                "tfidf",
                # just transform to scipy format
                FunctionTransformer(_tfidf_matrix),
                lambda frame: [name for name in tfidf if name in frame.columns],
            )
        )
    if vectors:
        # Resolve variable dimensions on train; keep missing E5 columns as errors.
        transformers.append(
            (
                "vectors",
                "passthrough",
                lambda frame: [
                    name
                    for name in vectors
                    if name not in TEXT_SVD_NUM or name in frame.columns
                ],
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=1.0)
