"""Features of the baseline, in three groups.

content    what the post is: length, hashtags, mentions, links, type, posting time
history    who wrote it: the author's level and spread, rhythm, follower count
artefact   how it was collected: the post's age at crawl and its place in the
           author's run
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

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
