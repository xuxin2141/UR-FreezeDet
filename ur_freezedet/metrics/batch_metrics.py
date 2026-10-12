"""Batch-grouped metric helpers."""

import pandas as pd


def batch_counts(frame):
    return frame.groupby("batch_id", dropna=False).size().rename("n").reset_index()
