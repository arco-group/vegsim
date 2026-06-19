"""Modular scenario-aware vegetation world model components."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PositionalEncoding


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    """Build a small feed-forward network."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers: list[nn.Module] = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _normal_sample(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _resolve_horizon_values(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    horizons: torch.Tensor | None,
) -> torch.Tensor:
    if horizons is None:
        values = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        return values

    if horizons.dim() == 1:
        if horizons.shape[0] != seq_len:
            raise ValueError("1D horizons must match sequence length")
        return horizons.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    if horizons.dim() == 2:
        if horizons.shape[1] != seq_len:
            raise ValueError("2D horizons must match sequence length")
        if horizons.shape[0] == 1 and batch_size > 1:
            return horizons.to(device=device, dtype=torch.float32).repeat(batch_size, 1)
        if horizons.shape[0] != batch_size:
            raise ValueError("2D horizons must match batch size")
        return horizons.to(device=device, dtype=torch.float32)

    raise ValueError("horizons must be None, [L], or [B,L]")


class SpatialEncoder(nn.Module):
    """Encode optional spatial continuous/categorical metadata.

    Inputs:
    - spatial_cont: [B, C_spatial_cont] or None
    - spatial_cat: [B, C_spatial_cat] or None

    Output:
    - spatial_emb: [B, D_spatial] or None when disabled
    """

    def __init__(
        self,
        enabled: bool,
        cont_dim: int,
        cat_cardinalities: list[int] | None,
        emb_dim: int,
        use_harmonic_coords: bool = False,
        num_harmonic_frequencies: int = 4,
        dropout: float = 0.1,
        cont_hidden_dim: int = 64,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.cont_dim = int(cont_dim)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.emb_dim = int(emb_dim)
        self.use_harmonic_coords = bool(use_harmonic_coords)
        self.num_harmonic_frequencies = int(num_harmonic_frequencies)

        self.cont_in_dim = self.cont_dim
        if self.use_harmonic_coords and self.cont_dim > 0:
            self.cont_in_dim += self.cont_dim * 2 * self.num_harmonic_frequencies

        self.cont_mlp = None
        if self.cont_dim > 0:
            self.cont_mlp = nn.Sequential(
                nn.Linear(self.cont_in_dim, cont_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(cont_hidden_dim, self.emb_dim),
            )

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(max(2, int(card)), self.emb_dim) for card in self.cat_cardinalities]
        )

        n_paths = 0
        if self.cont_mlp is not None:
            n_paths += 1
        n_paths += len(self.cat_embeddings)
        if n_paths == 0:
            self.output_proj = None
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(self.emb_dim * n_paths, self.emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def _harmonic_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_harmonic_coords or self.num_harmonic_frequencies <= 0:
            return x

        features = [x]
        for k in range(self.num_harmonic_frequencies):
            freq = 2.0**k
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        return torch.cat(features, dim=-1)

    def forward(
        self,
        spatial_cont: torch.Tensor | None,
        spatial_cat: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None

        chunks: list[torch.Tensor] = []

        if self.cont_mlp is not None:
            if spatial_cont is None:
                cont = torch.zeros(batch_size, self.cont_dim, device=device)
            else:
                cont = spatial_cont.to(device=device, dtype=torch.float32)
                if cont.dim() == 1:
                    cont = cont.unsqueeze(0)
                if cont.shape[0] != batch_size:
                    raise ValueError("spatial_cont batch dimension mismatch")
                if cont.shape[1] < self.cont_dim:
                    pad = torch.zeros(batch_size, self.cont_dim - cont.shape[1], device=device)
                    cont = torch.cat([cont, pad], dim=-1)
                elif cont.shape[1] > self.cont_dim:
                    cont = cont[:, : self.cont_dim]

            cont = torch.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
            cont_aug = self._harmonic_encode(cont)
            chunks.append(self.cont_mlp(cont_aug))

        if self.cat_embeddings:
            if spatial_cat is None:
                cat = torch.zeros(batch_size, len(self.cat_embeddings), device=device, dtype=torch.long)
            else:
                cat = spatial_cat.to(device=device, dtype=torch.long)
                if cat.dim() == 1:
                    cat = cat.unsqueeze(-1)
                if cat.shape[0] != batch_size:
                    raise ValueError("spatial_cat batch dimension mismatch")

            for idx, emb in enumerate(self.cat_embeddings):
                if idx < cat.shape[1]:
                    raw_idx = cat[:, idx]
                else:
                    raw_idx = torch.zeros(batch_size, device=device, dtype=torch.long)
                safe_idx = torch.remainder(torch.abs(raw_idx), emb.num_embeddings)
                chunks.append(emb(safe_idx))

        if not chunks:
            return torch.zeros(batch_size, self.emb_dim, device=device)

        if len(chunks) == 1:
            return chunks[0]
        return self.output_proj(torch.cat(chunks, dim=-1))


class HistoryEncoder(nn.Module):
    """Transformer history encoder.

    Inputs:
    - history: [B, T_hist, C_in]
    - history_mask: [B, T_hist, C_in]
    - history_pad_mask: [B, T_hist]

    Outputs:
    - hist_tokens: [B, T_hist, D]
    - hist_token_mask: [B, T_hist] (True means invalid token)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout)

    @staticmethod
    def _clean_sequence(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, torch.zeros_like(sequence), sequence)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        history_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = history_pad_mask | history_mask.all(dim=-1)
        clean = self._clean_sequence(history, history_mask)
        emb = self.pos(self.input_proj(clean))
        tokens = self.encoder(emb, src_key_padding_mask=token_mask)
        return tokens, token_mask


class AttentionPooling(nn.Module):
    """Learned-query attention pooling with masking."""

    def __init__(self, d_model: int, num_queries: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        # tokens [B, T, D], token_mask [B, T]
        scores = torch.einsum("qd,btd->bqt", self.query, tokens) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(token_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        valid = (~token_mask).unsqueeze(1)
        attn = attn * valid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = torch.einsum("bqt,btd->bqd", attn, tokens)
        return pooled.mean(dim=1)


class StateSummarizer(nn.Module):
    """Infer initial latent state from history tokens."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        summarizer_type: str = "attention_pooling",
        sample_initial_state: bool = True,
    ):
        super().__init__()
        self.summarizer_type = summarizer_type
        self.latent_dim = int(latent_dim)
        self.sample_initial_state = bool(sample_initial_state)

        self.attn_pool = AttentionPooling(d_model=d_model, num_queries=1)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def _masked_average(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        keep = (~token_mask).float()
        denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        summed = (tokens * keep.unsqueeze(-1)).sum(dim=1)
        return summed / denom

    def _pool(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if self.summarizer_type == "masked_average_pooling":
            return self._masked_average(tokens, token_mask)
        return self.attn_pool(tokens, token_mask)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, training: bool) -> dict[str, torch.Tensor]:
        pooled = self._pool(tokens, token_mask)

        if self.summarizer_type == "posterior_attention_pooling":
            mu = self.to_mu(pooled)
            logvar = self.to_logvar(pooled)
            z0 = _normal_sample(mu, logvar, sample=training and self.sample_initial_state)
            return {"z0": z0, "z0_mu": mu, "z0_logvar": logvar}

        z0 = self.to_latent(pooled)
        return {"z0": z0}


class HorizonEmbedding(nn.Module):
    """Horizon/lead-time embeddings."""

    def __init__(self, emb_dim: int, embedding_type: str = "learned", max_horizon: int = 512):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.embedding_type = embedding_type
        self.max_horizon = int(max_horizon)

        if embedding_type == "learned":
            self.learned = nn.Embedding(self.max_horizon, self.emb_dim)
            self.mlp = None
        elif embedding_type == "mlp":
            self.learned = None
            self.mlp = nn.Sequential(nn.Linear(1, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        elif embedding_type == "sinusoidal":
            self.learned = None
            self.mlp = None
        else:
            raise ValueError(f"Unsupported horizon_embedding_type: {embedding_type}")

    def _sinusoidal(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, L]
        bsz, seq_len = values.shape
        half = self.emb_dim // 2
        if half == 0:
            return values.unsqueeze(-1)
        div = torch.exp(
            torch.arange(0, half, device=values.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
        )
        phase = values.unsqueeze(-1) * div
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            pad = torch.zeros(bsz, seq_len, self.emb_dim - emb.shape[-1], device=values.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        horizons: torch.Tensor | None,
    ) -> torch.Tensor:
        values = _resolve_horizon_values(batch_size, seq_len, device, horizons)

        if self.embedding_type == "learned":
            idx = values.round().long().clamp(min=1) - 1
            idx = idx.clamp(max=self.max_horizon - 1)
            return self.learned(idx)

        if self.embedding_type == "mlp":
            return self.mlp(values.unsqueeze(-1))

        return self._sinusoidal(values)


class FutureEncoder(nn.Module):
    """Transformer future encoder used as exogenous driver encoder."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_horizon_embedding: bool = True,
        horizon_emb_dim: int = 32,
        horizon_embedding_type: str = "learned",
        max_horizon: int = 512,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.use_horizon_embedding = bool(use_horizon_embedding)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.horizon_embedding = None
        self.horizon_to_model = None
        self.horizon_emb_dim = 0
        if self.use_horizon_embedding:
            self.horizon_emb_dim = int(horizon_emb_dim)
            self.horizon_embedding = HorizonEmbedding(
                emb_dim=self.horizon_emb_dim,
                embedding_type=horizon_embedding_type,
                max_horizon=max_horizon,
            )
            self.horizon_to_model = nn.Linear(self.horizon_emb_dim, d_model)

    @staticmethod
    def _clean_sequence(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, torch.zeros_like(sequence), sequence)

    def forward(
        self,
        future: torch.Tensor,
        future_mask: torch.Tensor,
        future_pad_mask: torch.Tensor,
        horizons: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        token_mask = future_pad_mask | future_mask.all(dim=-1)
        clean = self._clean_sequence(future, future_mask)
        emb = self.input_proj(clean)

        horizon_emb = None
        if self.horizon_embedding is not None:
            horizon_emb = self.horizon_embedding(
                batch_size=future.shape[0],
                seq_len=future.shape[1],
                device=future.device,
                horizons=horizons,
            )
            emb = emb + self.horizon_to_model(horizon_emb)

        emb = self.pos(emb)
        tokens = self.encoder(emb, src_key_padding_mask=token_mask)
        return tokens, token_mask, horizon_emb


@dataclass
class DynamicsOutput:
    latent_states: torch.Tensor
    prior_mu: torch.Tensor | None = None
    prior_logvar: torch.Tensor | None = None
    posterior_mu: torch.Tensor | None = None
    posterior_logvar: torch.Tensor | None = None


class ResidualMLPDynamics(nn.Module):
    """Deterministic residual MLP latent dynamics."""

    def __init__(
        self,
        latent_dim: int,
        future_dim: int,
        spatial_dim: int,
        horizon_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        in_dim = latent_dim + future_dim + spatial_dim + horizon_dim
        self.delta_net = build_mlp(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        z0: torch.Tensor,
        future_tokens: torch.Tensor,
        spatial_emb: torch.Tensor | None,
        horizon_emb: torch.Tensor | None,
        future_pad_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> DynamicsOutput:
        bsz, seq_len, _ = future_tokens.shape
        z_prev = z0
        states = []

        for t in range(seq_len):
            parts = [z_prev, future_tokens[:, t]]
            if spatial_emb is not None:
                parts.append(spatial_emb)
            if horizon_emb is not None:
                parts.append(horizon_emb[:, t])

            delta = self.delta_net(torch.cat(parts, dim=-1))
            z_next = z_prev + delta

            valid = (~future_pad_mask[:, t]).unsqueeze(-1)
            z_next = torch.where(valid, z_next, z_prev)

            states.append(z_next)
            z_prev = z_next

        return DynamicsOutput(latent_states=torch.stack(states, dim=1))


class GRULatentDynamics(nn.Module):
    """GRU-based deterministic latent dynamics."""

    def __init__(
        self,
        latent_dim: int,
        future_dim: int,
        spatial_dim: int,
        horizon_dim: int,
        hidden_dim: int,
        dropout: float,
        use_residual_update: bool,
    ):
        super().__init__()
        in_dim = future_dim + spatial_dim + horizon_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cell = nn.GRUCell(hidden_dim, latent_dim)
        self.use_residual_update = bool(use_residual_update)
        self.residual_update = nn.Linear(latent_dim, latent_dim) if self.use_residual_update else None

    def forward(
        self,
        z0: torch.Tensor,
        future_tokens: torch.Tensor,
        spatial_emb: torch.Tensor | None,
        horizon_emb: torch.Tensor | None,
        future_pad_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> DynamicsOutput:
        bsz, seq_len, _ = future_tokens.shape
        z_prev = z0
        states = []

        for t in range(seq_len):
            parts = [future_tokens[:, t]]
            if spatial_emb is not None:
                parts.append(spatial_emb)
            if horizon_emb is not None:
                parts.append(horizon_emb[:, t])
            u_t = self.input_proj(torch.cat(parts, dim=-1))

            z_next = self.cell(u_t, z_prev)
            if self.use_residual_update:
                z_next = z_prev + self.residual_update(z_next)

            valid = (~future_pad_mask[:, t]).unsqueeze(-1)
            z_next = torch.where(valid, z_next, z_prev)

            states.append(z_next)
            z_prev = z_next

        return DynamicsOutput(latent_states=torch.stack(states, dim=1))


class TransformerLatentDynamics(nn.Module):
    """Causal-transformer deterministic latent dynamics."""

    def __init__(
        self,
        latent_dim: int,
        future_dim: int,
        spatial_dim: int,
        horizon_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_residual_update: bool,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 for transformer dynamics")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0 for transformer dynamics")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("transformer dynamics hidden_dim must be divisible by num_heads")

        exog_dim = future_dim + spatial_dim + horizon_dim
        self.input_proj = nn.Linear(exog_dim, self.hidden_dim)
        self.z0_proj = nn.Linear(latent_dim, self.hidden_dim)
        self.pos = PositionalEncoding(d_model=self.hidden_dim, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.to_delta = nn.Linear(self.hidden_dim, latent_dim)
        self.use_residual_update = bool(use_residual_update)
        self.residual_update = nn.Linear(latent_dim, latent_dim) if self.use_residual_update else None

    def forward(
        self,
        z0: torch.Tensor,
        future_tokens: torch.Tensor,
        spatial_emb: torch.Tensor | None,
        horizon_emb: torch.Tensor | None,
        future_pad_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> DynamicsOutput:
        bsz, seq_len, _ = future_tokens.shape

        parts = [future_tokens]
        if spatial_emb is not None:
            parts.append(spatial_emb.unsqueeze(1).expand(-1, seq_len, -1))
        if horizon_emb is not None:
            parts.append(horizon_emb)
        exog = torch.cat(parts, dim=-1)

        tokens = self.input_proj(exog) + self.z0_proj(z0).unsqueeze(1)
        tokens = self.pos(tokens)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=future_pad_mask,
        )
        deltas = self.to_delta(encoded)

        z_prev = z0
        states = []
        for t in range(seq_len):
            delta_t = deltas[:, t]
            if self.use_residual_update:
                z_next = z_prev + self.residual_update(delta_t)
            else:
                z_next = z_prev + delta_t

            valid = (~future_pad_mask[:, t]).unsqueeze(-1)
            z_next = torch.where(valid, z_next, z_prev)

            states.append(z_next)
            z_prev = z_next

        return DynamicsOutput(latent_states=torch.stack(states, dim=1))


class RSSMLatentDynamics(nn.Module):
    """RSSM-like stochastic latent dynamics.

    Posterior is only used where supervision is available.
    """

    def __init__(
        self,
        latent_dim: int,
        future_dim: int,
        spatial_dim: int,
        horizon_dim: int,
        y_dim: int,
        deterministic_dim: int,
        stochastic_dim: int,
        hidden_dim: int,
        dropout: float,
        sample_latent: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.stochastic_dim = int(stochastic_dim)
        self.det_dim = int(deterministic_dim)
        self.sample_latent = bool(sample_latent)

        exog_dim = future_dim + spatial_dim + horizon_dim
        self.exog_dim = exog_dim

        self.z0_to_stoch = nn.Linear(latent_dim, self.stochastic_dim)
        self.stoch_to_latent = nn.Linear(self.stochastic_dim, latent_dim)
        self.stoch_to_det0 = nn.Linear(self.stochastic_dim, self.det_dim)

        self.prior_net = build_mlp(
            input_dim=self.det_dim + exog_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * self.stochastic_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.posterior_net = build_mlp(
            input_dim=self.det_dim + exog_dim + y_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * self.stochastic_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.det_cell = nn.GRUCell(self.stochastic_dim + exog_dim, self.det_dim)

    def forward(
        self,
        z0: torch.Tensor,
        future_tokens: torch.Tensor,
        spatial_emb: torch.Tensor | None,
        horizon_emb: torch.Tensor | None,
        future_pad_mask: torch.Tensor,
        y_future: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        posterior_use_probability: float | None = None,
        training: bool = True,
        **_: torch.Tensor,
    ) -> DynamicsOutput:
        bsz, seq_len, _ = future_tokens.shape

        z_prev_stoch = self.z0_to_stoch(z0)
        h_prev = self.stoch_to_det0(z_prev_stoch)

        latent_states: list[torch.Tensor] = []
        prior_mus: list[torch.Tensor] = []
        prior_logvars: list[torch.Tensor] = []
        post_mus: list[torch.Tensor] = []
        post_logvars: list[torch.Tensor] = []

        for t in range(seq_len):
            exog_parts = [future_tokens[:, t]]
            if spatial_emb is not None:
                exog_parts.append(spatial_emb)
            if horizon_emb is not None:
                exog_parts.append(horizon_emb[:, t])
            exog_t = torch.cat(exog_parts, dim=-1)

            prior_stats = self.prior_net(torch.cat([h_prev, exog_t], dim=-1))
            prior_mu, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)
            z_prior = _normal_sample(prior_mu, prior_logvar, sample=training and self.sample_latent)

            z_used = z_prior
            post_mu, post_logvar = prior_mu, prior_logvar

            if training and y_future is not None:
                y_t = y_future[:, t]
                post_stats = self.posterior_net(torch.cat([h_prev, exog_t, y_t], dim=-1))
                cand_mu, cand_logvar = torch.chunk(post_stats, 2, dim=-1)
                z_post = _normal_sample(cand_mu, cand_logvar, sample=self.sample_latent)

                if future_target_mask is None:
                    use_post = torch.ones(bsz, 1, dtype=torch.bool, device=z_post.device)
                else:
                    # mask=True means invalid. Use posterior when at least one valid target channel exists.
                    use_post = (~future_target_mask[:, t]).any(dim=-1, keepdim=True)

                if posterior_use_probability is not None:
                    p_post = float(max(0.0, min(1.0, posterior_use_probability)))
                    if p_post <= 0.0:
                        use_post = torch.zeros_like(use_post)
                    elif p_post < 1.0:
                        draw = torch.rand(bsz, 1, device=z_post.device)
                        use_post = use_post & (draw < p_post)

                z_used = torch.where(use_post, z_post, z_prior)
                post_mu = torch.where(use_post, cand_mu, prior_mu)
                post_logvar = torch.where(use_post, cand_logvar, prior_logvar)

            h_next = self.det_cell(torch.cat([z_used, exog_t], dim=-1), h_prev)

            valid = (~future_pad_mask[:, t]).unsqueeze(-1)
            z_used = torch.where(valid, z_used, z_prev_stoch)
            h_next = torch.where(valid, h_next, h_prev)

            z_prev_stoch = z_used
            h_prev = h_next

            latent_states.append(self.stoch_to_latent(z_used))
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
            post_mus.append(post_mu)
            post_logvars.append(post_logvar)

        return DynamicsOutput(
            latent_states=torch.stack(latent_states, dim=1),
            prior_mu=torch.stack(prior_mus, dim=1),
            prior_logvar=torch.stack(prior_logvars, dim=1),
            posterior_mu=torch.stack(post_mus, dim=1),
            posterior_logvar=torch.stack(post_logvars, dim=1),
        )


class QuantileDecoder(nn.Module):
    """Decode latent states into ordered/unordered target quantiles."""

    def __init__(
        self,
        latent_dim: int,
        target_dim: int,
        num_quantiles: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_horizon_conditioned_decoder: bool,
        horizon_dim: int,
        spatial_dim: int,
        enforce_quantile_ordering: bool,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.num_quantiles = int(num_quantiles)
        self.use_horizon_conditioned_decoder = bool(use_horizon_conditioned_decoder)
        self.enforce_quantile_ordering = bool(enforce_quantile_ordering)

        in_dim = latent_dim
        if self.use_horizon_conditioned_decoder:
            in_dim += horizon_dim
        if spatial_dim > 0:
            in_dim += spatial_dim

        self.net = build_mlp(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            output_dim=self.target_dim * self.num_quantiles,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        latent_states: torch.Tensor,
        horizon_emb: torch.Tensor | None,
        spatial_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = latent_states.shape

        parts = [latent_states]
        if self.use_horizon_conditioned_decoder and horizon_emb is not None:
            parts.append(horizon_emb)
        if spatial_emb is not None:
            parts.append(spatial_emb.unsqueeze(1).expand(-1, seq_len, -1))

        x = torch.cat(parts, dim=-1)
        raw = self.net(x).view(bsz, seq_len, self.target_dim, self.num_quantiles)

        if not self.enforce_quantile_ordering or self.num_quantiles <= 1:
            return raw

        q0 = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        # Avoid torch.cumsum here because CUDA deterministic mode can reject it.
        ordered = [q0]
        current = q0
        for idx in range(increments.shape[-1]):
            current = current + increments[..., idx : idx + 1]
            ordered.append(current)
        return torch.cat(ordered, dim=-1)


class RiskHead(nn.Module):
    """Optional risk head for stress probability logits."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.net(latent_states)


class VegetationWorldModel(nn.Module):
    """Scenario-aware latent world model for vegetation forecasting.

    Inputs expected in batch:
    - history: [B, T_hist, C_in]
    - history_mask: [B, T_hist, C_in]
    - history_pad_mask: [B, T_hist]
    - future: [B, L, C_in]
    - future_mask: [B, L, C_in]
    - future_pad_mask: [B, L]
    - future_delta_days or horizons: [B, L] or [L] (optional)
    - spatial_cont: [B, C_spatial_cont] (optional)
    - spatial_cat: [B, C_spatial_cat] (optional)
    - target_dense: [B, L, C_y] (optional, for RSSM posterior and losses)
    - target_dense_mask: [B, L, C_y] (optional)

    Outputs:
    - quantiles: [B, L, C_y, Q]
    - latent_states: [B, L, D_latent]
    - optional risk_logits: [B, L, C_y]
    - optional prior/posterior params for RSSM
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        quantiles: list[float],
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        summarizer_type: str = "attention_pooling",
        latent_dim: int = 128,
        sample_initial_state: bool = True,
        use_horizon_embedding: bool = True,
        horizon_emb_dim: int = 32,
        horizon_embedding_type: str = "learned",
        max_horizon: int = 512,
        spatial_enabled: bool = False,
        spatial_cont_dim: int = 0,
        spatial_cat_cardinalities: list[int] | None = None,
        spatial_emb_dim: int = 32,
        use_harmonic_coords: bool = False,
        num_harmonic_frequencies: int = 4,
        spatial_dropout: float = 0.1,
        dynamics_type: str = "gru",
        dynamics_hidden_dim: int = 128,
        dynamics_num_layers: int = 2,
        use_residual_update: bool = False,
        rssm_deterministic_dim: int = 128,
        rssm_stochastic_dim: int = 128,
        rssm_hidden_dim: int = 128,
        sample_latent: bool = True,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        use_horizon_conditioned_decoder: bool = False,
        enforce_quantile_ordering: bool = False,
        use_risk_head: bool = False,
    ):
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles cannot be empty")
        if sorted(quantiles) != list(quantiles):
            raise ValueError("quantiles must be in increasing order")

        self.quantiles = list(quantiles)
        self.target_dim = int(target_dim)
        self.latent_dim = int(latent_dim)
        self.dynamics_type = dynamics_type

        self.history_encoder = HistoryEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.summarizer = StateSummarizer(
            d_model=d_model,
            latent_dim=latent_dim,
            summarizer_type=summarizer_type,
            sample_initial_state=sample_initial_state,
        )
        self.future_encoder = FutureEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_horizon_embedding=use_horizon_embedding,
            horizon_emb_dim=horizon_emb_dim,
            horizon_embedding_type=horizon_embedding_type,
            max_horizon=max_horizon,
        )

        self.spatial_encoder = SpatialEncoder(
            enabled=spatial_enabled,
            cont_dim=spatial_cont_dim,
            cat_cardinalities=spatial_cat_cardinalities,
            emb_dim=spatial_emb_dim,
            use_harmonic_coords=use_harmonic_coords,
            num_harmonic_frequencies=num_harmonic_frequencies,
            dropout=spatial_dropout,
        )
        spatial_dim = spatial_emb_dim if spatial_enabled else 0
        horizon_dim = self.future_encoder.horizon_emb_dim if use_horizon_embedding else 0

        if dynamics_type == "residual_mlp":
            self.dynamics = ResidualMLPDynamics(
                latent_dim=latent_dim,
                future_dim=d_model,
                spatial_dim=spatial_dim,
                horizon_dim=horizon_dim,
                hidden_dim=dynamics_hidden_dim,
                num_layers=dynamics_num_layers,
                dropout=dropout,
            )
        elif dynamics_type == "gru":
            self.dynamics = GRULatentDynamics(
                latent_dim=latent_dim,
                future_dim=d_model,
                spatial_dim=spatial_dim,
                horizon_dim=horizon_dim,
                hidden_dim=dynamics_hidden_dim,
                dropout=dropout,
                use_residual_update=use_residual_update,
            )
        elif dynamics_type == "transformer":
            self.dynamics = TransformerLatentDynamics(
                latent_dim=latent_dim,
                future_dim=d_model,
                spatial_dim=spatial_dim,
                horizon_dim=horizon_dim,
                hidden_dim=dynamics_hidden_dim,
                num_layers=dynamics_num_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_residual_update=use_residual_update,
            )
        elif dynamics_type == "rssm":
            self.dynamics = RSSMLatentDynamics(
                latent_dim=latent_dim,
                future_dim=d_model,
                spatial_dim=spatial_dim,
                horizon_dim=horizon_dim,
                y_dim=target_dim,
                deterministic_dim=rssm_deterministic_dim,
                stochastic_dim=rssm_stochastic_dim,
                hidden_dim=rssm_hidden_dim,
                dropout=dropout,
                sample_latent=sample_latent,
            )
        else:
            raise ValueError(f"Unsupported dynamics_type: {dynamics_type}")

        self.decoder = QuantileDecoder(
            latent_dim=latent_dim,
            target_dim=target_dim,
            num_quantiles=len(self.quantiles),
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            use_horizon_conditioned_decoder=use_horizon_conditioned_decoder,
            horizon_dim=horizon_dim,
            spatial_dim=spatial_dim,
            enforce_quantile_ordering=enforce_quantile_ordering,
        )

        self.risk_head = RiskHead(
            latent_dim=latent_dim,
            output_dim=target_dim,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        ) if use_risk_head else None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        y_future_override: torch.Tensor | None = None,
        posterior_use_probability: float | None = None,
    ) -> dict[str, torch.Tensor]:
        history = batch["history"]
        history_mask = batch["history_mask"]
        history_pad_mask = batch["history_pad_mask"]

        future = batch["future"]
        future_mask = batch["future_mask"]
        future_pad_mask = batch["future_pad_mask"]

        horizons = batch.get("horizons", batch.get("future_delta_days", None))

        hist_tokens, hist_token_mask = self.history_encoder(history, history_mask, history_pad_mask)
        summary = self.summarizer(hist_tokens, hist_token_mask, training=self.training)
        z0 = summary["z0"]

        future_tokens, _, horizon_emb = self.future_encoder(
            future,
            future_mask,
            future_pad_mask,
            horizons=horizons,
        )

        spatial_emb = self.spatial_encoder(
            spatial_cont=batch.get("spatial_cont", None),
            spatial_cat=batch.get("spatial_cat", None),
            batch_size=history.shape[0],
            device=history.device,
        )

        y_future = y_future_override if y_future_override is not None else batch.get("target_dense", None)
        future_target_mask = batch.get("target_dense_mask", None)

        dyn_out = self.dynamics(
            z0=z0,
            future_tokens=future_tokens,
            spatial_emb=spatial_emb,
            horizon_emb=horizon_emb,
            y_future=y_future,
            future_target_mask=future_target_mask,
            posterior_use_probability=posterior_use_probability,
            training=self.training,
            future_pad_mask=future_pad_mask,
        )

        quantiles = self.decoder(dyn_out.latent_states, horizon_emb=horizon_emb, spatial_emb=spatial_emb)

        out = {
            "quantiles": quantiles,
            "latent_states": dyn_out.latent_states,
            "z0": z0,
            "hist_tokens": hist_tokens,
            "hist_token_mask": hist_token_mask,
        }
        out.update(summary)

        if dyn_out.prior_mu is not None:
            out["prior_mu"] = dyn_out.prior_mu
            out["prior_logvar"] = dyn_out.prior_logvar
            out["posterior_mu"] = dyn_out.posterior_mu
            out["posterior_logvar"] = dyn_out.posterior_logvar

        if self.risk_head is not None:
            out["risk_logits"] = self.risk_head(dyn_out.latent_states)

        return out


class SketchedIsotropicGaussianRegularizer(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (SIGReg).

    This module implements an Epps-Pulley-based distribution matching statistic
    over random one-dimensional projections.

    Expected input:
    - proj: [N, D] or [T, B, D]
    """

    def __init__(self, knots: int = 17, num_proj: int = 256, t_max: float = 3.0, eps: float = 1e-8):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be >= 2")
        if num_proj < 1:
            raise ValueError("num_proj must be >= 1")

        self.num_proj = int(num_proj)
        self.eps = float(eps)

        t = torch.linspace(0.0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / float(knots - 1)
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() == 3:
            x = proj.reshape(-1, proj.shape[-1])
        elif proj.dim() == 2:
            x = proj
        else:
            raise ValueError("proj must have shape [N, D] or [T, B, D]")

        if x.shape[0] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        x = x.float()

        # Random unit-norm projections sampled on the hypersphere.
        a = torch.randn(x.shape[-1], self.num_proj, device=x.device, dtype=x.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp(min=self.eps)

        t = self.t.to(device=x.device, dtype=x.dtype)
        phi = self.phi.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)

        x_t = (x @ a).unsqueeze(-1) * t  # [N, P, K]
        err = (x_t.cos().mean(dim=0) - phi).square() + x_t.sin().mean(dim=0).square()  # [P, K]
        statistic = (err @ weights) * x.shape[0]  # [P]
        return statistic.mean()


def weighted_quantile_loss_dense(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    quantiles: list[float],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted pinball loss for dense predictions.

    Shapes:
    - preds: [B, L, C, Q]
    - targets: [B, L, C]
    - mask: [B, L, C] (True means invalid)
    - weights: [L] or [B, L] or [B, L, C]
    """

    if preds.shape[:-1] != targets.shape:
        raise ValueError("preds and targets shapes are incompatible")
    if targets.shape != mask.shape:
        raise ValueError("targets and mask shapes must match")
    if preds.shape[-1] != len(quantiles):
        raise ValueError("preds quantile dim mismatch")

    keep = (~mask).float()
    if weights is not None:
        if weights.dim() == 1:
            weights = weights.view(1, -1, 1)
        elif weights.dim() == 2:
            weights = weights.unsqueeze(-1)
        elif weights.dim() == 3:
            pass
        else:
            raise ValueError("weights must be 1D, 2D, or 3D")
        keep = keep * weights.to(device=preds.device, dtype=preds.dtype)

    target_q = targets.unsqueeze(-1)
    diff = target_q - preds
    tau = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype).view(1, 1, 1, -1)
    pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)
    weighted = pinball * keep.unsqueeze(-1)

    denom = keep.sum().clamp(min=1e-8) * len(quantiles)
    return weighted.sum() / denom


def masked_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    keep = ~mask
    if not keep.any():
        return torch.tensor(0.0, device=preds.device)

    diff = preds[keep] - targets[keep]
    if loss_type == "mae":
        return diff.abs().mean()
    if loss_type == "mse":
        return (diff * diff).mean()
    if loss_type == "huber":
        return F.huber_loss(preds[keep], targets[keep], reduction="mean")
    raise ValueError(f"Unsupported aux loss type: {loss_type}")


def latent_smoothness_loss(
    latent_states: torch.Tensor,
    future_pad_mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if latent_states.shape[1] < 2:
        return torch.tensor(0.0, device=latent_states.device)

    diffs = latent_states[:, 1:] - latent_states[:, :-1]
    valid = ((~future_pad_mask[:, 1:]) & (~future_pad_mask[:, :-1])).unsqueeze(-1)
    if not valid.any():
        return torch.tensor(0.0, device=latent_states.device)

    diffs = diffs.masked_select(valid).view(-1)
    if loss_type == "l2":
        return (diffs * diffs).mean()
    if loss_type == "huber":
        return F.huber_loss(diffs, torch.zeros_like(diffs), reduction="mean")
    raise ValueError(f"Unsupported smoothness loss type: {loss_type}")


def kl_normal(prior_mu: torch.Tensor, prior_logvar: torch.Tensor, post_mu: torch.Tensor, post_logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians."""
    return 0.5 * (
        prior_logvar
        - post_logvar
        + (torch.exp(post_logvar) + (post_mu - prior_mu) ** 2) / torch.exp(prior_logvar)
        - 1.0
    )


def masked_kl_loss(
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    post_mu: torch.Tensor,
    post_logvar: torch.Tensor,
    future_target_mask: torch.Tensor,
    free_nats: float = 0.0,
) -> torch.Tensor:
    kl = kl_normal(prior_mu, prior_logvar, post_mu, post_logvar).mean(dim=-1)
    valid = (~future_target_mask).any(dim=-1).float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=prior_mu.device)

    if free_nats > 0.0:
        kl = torch.clamp(kl, min=float(free_nats))

    return (kl * valid).sum() / valid.sum().clamp(min=1.0)


def noncrossing_loss(preds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if preds.shape[-1] <= 1:
        return torch.tensor(0.0, device=preds.device)
    penalty = torch.relu(preds[..., :-1] - preds[..., 1:])
    keep = (~mask).float().unsqueeze(-1)
    denom = keep.sum().clamp(min=1e-8) * (preds.shape[-1] - 1)
    return (penalty * keep).sum() / denom


def masked_mae_rmse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = ~mask
    if not keep.any():
        zero = torch.tensor(0.0, device=preds.device)
        return zero, zero
    diff = preds[keep] - targets[keep]
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff * diff).mean() + 1e-12)
    return mae, rmse


def calibration_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    quantiles: list[float],
) -> dict[str, torch.Tensor]:
    keep = ~mask
    out: dict[str, torch.Tensor] = {}
    if not keep.any():
        zero = torch.tensor(0.0, device=preds.device)
        out["coverage"] = zero
        out["interval_width"] = zero
        out["calibration_error"] = zero
        return out

    if preds.shape[-1] >= 2:
        low = preds[..., 0]
        high = preds[..., -1]
        inside = ((targets >= low) & (targets <= high) & keep).float()
        coverage = inside.sum() / keep.sum().clamp(min=1.0)
        width = ((high - low).abs() * keep.float()).sum() / keep.sum().clamp(min=1.0)
    else:
        coverage = torch.tensor(0.0, device=preds.device)
        width = torch.tensor(0.0, device=preds.device)

    errors = []
    for idx, tau in enumerate(quantiles):
        event = (targets <= preds[..., idx]) & keep
        empirical = event.float().sum() / keep.sum().clamp(min=1.0)
        errors.append(torch.abs(empirical - torch.tensor(float(tau), device=preds.device)))
    cal_err = torch.stack(errors).mean() if errors else torch.tensor(0.0, device=preds.device)

    out["coverage"] = coverage
    out["interval_width"] = width
    out["calibration_error"] = cal_err
    return out
