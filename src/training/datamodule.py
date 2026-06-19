"""Lightning DataModule for cached time-series data."""

from __future__ import annotations

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import CacheTimeSeriesDataset
from training.batch import collate_variable


class CacheDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_root: str,
        scaler_path: str | None = None,
        val_cache_root: str | None = None,
        test_cache_root: str | None = None,
        predict_cache_root: str | None = None,
        batch_size: int = 64,
        train_split: float = 0.8,
        seed: int = 42,
        num_workers: int = 0,
        apply_scaling: bool = True,
        data_summary_path: str = "Data/dataSummary_completed.csv",
        min_crop_pixels: float = 0.0,
        feature_engineering: bool = False,
        discretize_target: bool = False,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.cache_root = cache_root
        self.scaler_path = scaler_path
        self.val_cache_root = val_cache_root
        self.test_cache_root = test_cache_root
        self.predict_cache_root = predict_cache_root
        self.batch_size = batch_size
        self.train_split = train_split
        self.seed = seed
        self.num_workers = num_workers
        self.apply_scaling = apply_scaling
        self.data_summary_path = data_summary_path
        self.min_crop_pixels = min_crop_pixels
        self.feature_engineering = feature_engineering
        self.discretize_target = discretize_target
        self.pin_memory = pin_memory

        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = CacheTimeSeriesDataset(
                cache_dir=self.cache_root,
                apply_scaling=self.apply_scaling,
                scaler_path=self.scaler_path,
                data_summary_path=self.data_summary_path,
                min_crop_pixels=self.min_crop_pixels,
                feature_engineering=self.feature_engineering,
                discretize_target=self.discretize_target,
            )

            if self.val_cache_root:
                self.train_set = self.dataset
                self.val_set = CacheTimeSeriesDataset(
                    cache_dir=self.val_cache_root,
                    apply_scaling=self.apply_scaling,
                    scaler_path=self.scaler_path,
                    data_summary_path=self.data_summary_path,
                    min_crop_pixels=self.min_crop_pixels,
                    feature_engineering=self.feature_engineering,
                    discretize_target=self.discretize_target,
                )
            else:
                total = len(self.dataset)
                if total < 2:
                    raise ValueError(
                        f"Dataset has {total} sample(s). At least 2 samples are required for train/val split."
                    )
                train_size = int(round(total * self.train_split))
                train_size = max(1, min(total - 1, train_size))
                val_size = total - train_size
                self.train_set, self.val_set = random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

            if self.test_cache_root:
                self.test_set = CacheTimeSeriesDataset(
                    cache_dir=self.test_cache_root,
                    apply_scaling=self.apply_scaling,
                    scaler_path=self.scaler_path,
                    data_summary_path=self.data_summary_path,
                    min_crop_pixels=self.min_crop_pixels,
                    feature_engineering=self.feature_engineering,
                    discretize_target=self.discretize_target,
                )

            if self.predict_cache_root:
                self.predict_set = CacheTimeSeriesDataset(
                    cache_dir=self.predict_cache_root,
                    apply_scaling=self.apply_scaling,
                    scaler_path=self.scaler_path,
                    data_summary_path=self.data_summary_path,
                    min_crop_pixels=self.min_crop_pixels,
                    feature_engineering=self.feature_engineering,
                    discretize_target=self.discretize_target,
                )
            elif self.test_set is not None:
                self.predict_set = self.test_set

    @property
    def feature_names(self):
        if self.dataset is None:
            return None
        return self.dataset.feature_names

    def get_spatial_cat_cardinalities(self) -> list[int]:
        if self.dataset is None:
            return [128, 128]
        climate_card = max(2, len(set(self.dataset.climates)))
        crop_card = max(2, len(set(self.dataset.crop_types)))
        return [climate_card, crop_card]

    def get_target_dim(self) -> int:
        if self.dataset is None:
            return 1
        return 1 if self.dataset.targets.ndim == 2 else int(self.dataset.targets.shape[-1])

    def get_input_dim(self) -> int:
        if self.dataset is None or not self.dataset.feature_names:
            return 0
        return len(self.dataset.feature_names)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_variable,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_variable,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_set is None:
            return None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_variable,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        if self.predict_set is None:
            return None
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_variable,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
