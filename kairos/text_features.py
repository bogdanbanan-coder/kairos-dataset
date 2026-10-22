"""Fixed-size text representations shared by the classifiers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TFIDF_MAX_FEATURES = 1024
SVD_COMPONENTS = 128


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
            for name in ("modules.json", "config.json", "tokenizer.json", "model.safetensors")
        ),
    )
    model.max_seq_length = 512
    indices = [i for i, text in enumerate(texts) if text.strip()]
    embeddings = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
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


def build(train_text: list[str], test_text: list[str], results_dir: Path) -> dict:
    """Fit corpus-based encoders on train only and encode train and test posts."""
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    tfidf_train = vectorizer.fit_transform(train_text)
    tfidf_test = vectorizer.transform(test_text)

    lsa = make_pipeline(
        TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42),
        Normalizer(),
    )
    svd_train = lsa.fit_transform(tfidf_train).astype(np.float32)
    svd_test = lsa.transform(tfidf_test).astype(np.float32)

    e5 = _e5_embeddings(train_text + test_text, results_dir)
    n_train = len(train_text)
    return {
        "text_tf_idf": (tfidf_train, tfidf_test),
        "text_tf_idf_truncatedsvd": (svd_train, svd_test),
        "text_multilingual_e5_small": (e5[:n_train], e5[n_train:]),
    }
