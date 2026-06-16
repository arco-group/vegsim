"""Transformer quantile regressor for vegetation forecasting."""

import torch
import torch.nn as nn

from .layers import PositionalEncoding


class AgriMatNetQuantile(nn.Module):
    def __init__(
        self,
        input_dim,
        quantiles,
        d_model=128,
        num_layers=2,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        if not quantiles:
            raise ValueError("The quantiles list cannot be empty")
        if sorted(quantiles) != list(quantiles):
            raise ValueError("Quantiles must be provided in increasing order")
        if len(set(quantiles)) != len(quantiles):
            raise ValueError("Quantiles must be unique")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Invalid quantile: {q}")

        self.input_dim = input_dim
        self.quantiles = list(quantiles)
        self.d_model = d_model

        self.history_proj = nn.Linear(input_dim, d_model)
        self.future_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.future_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.history_pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.future_pos = PositionalEncoding(d_model=d_model, dropout=dropout)

        fusion_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.quantiles)),
        )

    @staticmethod
    def _clean_sequence(sequence, mask):
        zeros = torch.zeros_like(sequence)
        return torch.where(mask, zeros, sequence)

    def encode_history(self, history, history_mask, history_pad_mask):
        tokens_to_mask = history_pad_mask | history_mask.all(dim=-1)
        clean = self._clean_sequence(history, history_mask)
        emb = self.history_proj(clean)
        emb = self.history_pos(emb)
        encoded = self.history_encoder(emb, src_key_padding_mask=tokens_to_mask)
        keep = ~tokens_to_mask
        weights = keep.float()
        summed = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        summary = summed / counts.unsqueeze(-1)
        return encoded, summary, tokens_to_mask

    def encode_future(self, future, future_mask, future_pad_mask):
        tokens_to_mask = future_pad_mask | future_mask.all(dim=-1)
        clean = self._clean_sequence(future, future_mask)
        emb = self.future_proj(clean)
        emb = self.future_pos(emb)
        encoded = self.future_encoder(emb, src_key_padding_mask=tokens_to_mask)
        return encoded, tokens_to_mask

    def forward(self, batch):
        history = batch["history"]
        future = batch["future"]
        history_mask = batch["history_mask"]
        future_mask = batch["future_mask"]
        history_pad_mask = batch["history_pad_mask"]
        future_pad_mask = batch["future_pad_mask"]
        future_target_positions = batch["future_target_positions"]

        _, history_summary, _ = self.encode_history(history, history_mask, history_pad_mask)
        future_encoded, _ = self.encode_future(future, future_mask, future_pad_mask)

        forecast_window = future_target_positions.shape[1]
        gather_index = future_target_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        selected_future = torch.gather(future_encoded, 1, gather_index)

        history_context = history_summary.unsqueeze(1).expand(-1, forecast_window, -1)
        fusion = torch.cat([selected_future, history_context], dim=-1)
        return self.head(fusion)


def quantile_loss(preds, targets, mask, quantiles, weights=None, monotonicity_weight: float = 0.0):
    if preds.size(-1) != len(quantiles):
        raise ValueError("Last dimension of preds does not match quantiles length")

    keep = ~mask
    weight_tensor = keep.float()
    if weights is not None:
        if weights.shape != keep.shape:
            raise ValueError("weights must have same shape as targets/mask")
        weight_tensor = weight_tensor * weights

    targets = targets.unsqueeze(-1)
    diff = targets - preds

    losses = []
    for idx, tau in enumerate(quantiles):
        diff_tau = diff[..., idx]
        positive = (diff_tau >= 0).float()
        loss_tau = torch.abs(diff_tau) * (tau * positive + (1 - tau) * (1 - positive))
        losses.append(loss_tau)

    loss_stack = torch.stack(losses, dim=-1)
    loss_stack = loss_stack * weight_tensor.unsqueeze(-1)
    denom = weight_tensor.sum().clamp(min=1e-8).float()
    base_loss = loss_stack.sum() / (denom * len(quantiles))

    if monotonicity_weight > 0.0 and preds.size(-1) > 1:
        diffs = preds[..., :-1] - preds[..., 1:]
        monotonicity_penalty = torch.relu(diffs)
        monotonicity_penalty = monotonicity_penalty * weight_tensor.unsqueeze(-1)
        monotonicity_penalty = monotonicity_penalty.sum() / (denom * (preds.size(-1) - 1))
        return base_loss + monotonicity_weight * monotonicity_penalty

    return base_loss
