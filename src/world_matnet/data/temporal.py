"""Temporal feature utilities."""

import numpy as np
import pandas as pd


def cyclical_encoding(values, period, encoding_type):
    if encoding_type not in {"sin", "cos"}:
        raise ValueError("encoding_type must be 'sin' or 'cos'")

    angle = 2 * np.pi * (np.asarray(values) % period) / period
    if encoding_type == "sin":
        return np.sin(angle)
    return np.cos(angle)


def fourier_encoding(values, period, harmonics):
    features = {}
    base = (np.asarray(values) % period) / period
    for harmonic in range(1, harmonics + 1):
        angle = 2 * np.pi * harmonic * base
        features[f"fourier_sin_{harmonic}"] = np.sin(angle)
        features[f"fourier_cos_{harmonic}"] = np.cos(angle)
    return features


def add_temporal_features(
    dataframe,
    add_hour=False,
    add_day=False,
    add_month=False,
    add_day_of_year=True,
    fourier_harmonics=1,
):
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError("A DatetimeIndex is required to generate temporal features")

    enriched = dataframe.copy()
    index = enriched.index

    if add_hour:
        enriched["hour"] = index.hour
        enriched["hour_sin"] = cyclical_encoding(index.hour, 24, "sin")
        enriched["hour_cos"] = cyclical_encoding(index.hour, 24, "cos")

    if add_day:
        enriched["day"] = index.day
        enriched["day_sin"] = cyclical_encoding(index.day - 1, 31, "sin")
        enriched["day_cos"] = cyclical_encoding(index.day - 1, 31, "cos")

    if add_month:
        enriched["month"] = index.month
        enriched["month_sin"] = cyclical_encoding(index.month - 1, 12, "sin")
        enriched["month_cos"] = cyclical_encoding(index.month - 1, 12, "cos")

    if add_day_of_year:
        day_of_year = index.dayofyear
        enriched["day_of_year"] = day_of_year
        fourier = fourier_encoding(day_of_year - 1, 366, max(1, fourier_harmonics))
        for name, values in fourier.items():
            enriched[f"day_of_year_{name}"] = values

    return enriched
