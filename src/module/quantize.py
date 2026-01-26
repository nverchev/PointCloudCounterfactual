"""Vector quantization utilities."""

import torch

from src.config import Experiment
from src.utils.neighbour_ops import pykeops_square_distance


class VectorQuantizer:
    """Handles vector quantization operations."""

    def __init__(self):
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.book_size: int = cfg_ae_model.book_size
        self.momentum = 0.9
        return

    def quantize(self, x: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input using codebook."""
        batch, _ = x.size()
        x_flat = x.view(batch * self.n_codes, 1, self.embedding_dim)
        book_repeated = codebook.repeat(batch, 1, 1)
        dist = pykeops_square_distance(x_flat, book_repeated)
        idx_flat = dist.argmin(axis=2)
        idx = idx_flat.view(batch, self.n_codes)
        embeddings = self._get_embeddings(idx_flat, book_repeated)
        dist_sum = dist.sum(1).view(batch, self.n_codes, self.book_size)
        return embeddings, idx, dist_sum

    def _get_embeddings(self, idx_flat: torch.Tensor, book_repeated: torch.Tensor) -> torch.Tensor:
        """Get embeddings from indices (already concatenated)."""
        idx_expanded = idx_flat.expand(-1, -1, self.embedding_dim)
        embeddings = book_repeated.gather(1, idx_expanded)
        return embeddings.view(-1, self.n_codes * self.embedding_dim)

    def create_one_hot(self, idx: torch.Tensor) -> torch.Tensor:
        """Create one-hot encoding of indices."""
        batch = idx.shape[0]
        one_hot = torch.zeros(batch, self.n_codes, self.book_size, device=idx.device)
        return one_hot.scatter_(2, idx.view(batch, self.n_codes, 1), 1)

    def decode_from_indices(self, idx: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Decode embeddings from indices."""
        batch = idx.shape[0]
        book = codebook.repeat(batch, 1, 1)
        idx_flat = idx.view(batch * self.n_codes, 1, 1)
        idx_expanded = idx_flat.expand(-1, -1, self.embedding_dim)
        embeddings = book.gather(1, idx_expanded)
        return embeddings.view(batch, self.n_codes * self.embedding_dim)

    def update_codebook(self, codebook: torch.Tensor, idx: torch.Tensor, x: torch.Tensor) -> None:
        """Update the codebook using EMA without tracking counts."""
        batch = idx.shape[0]

        # Reshape x
        x_reshaped = x.view(batch, self.n_codes, self.embedding_dim)

        # Create flat indices that account for n_codes dimension
        # idx: [batch, n_codes] with values in [0, book_size)
        # We need unique indices for each (code_idx, book_idx) pair
        code_offsets = torch.arange(self.n_codes, device=idx.device).view(1, self.n_codes) * self.book_size
        idx_offset = idx + code_offsets  # [batch, n_codes]

        # Flatten
        idx_flat = idx_offset.view(-1)  # [batch * n_codes]
        x_flat = x_reshaped.view(-1, self.embedding_dim)  # [batch * n_codes, embedding_dim]

        # Initialize accumulators for all codes
        total_size = self.n_codes * self.book_size
        embed_sum = torch.zeros(total_size, self.embedding_dim, device=codebook.device)
        counts = torch.zeros(total_size, device=codebook.device)

        # Accumulate using scatter_add
        embed_sum.scatter_add_(0, idx_flat.unsqueeze(1).expand(-1, self.embedding_dim), x_flat)
        counts.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=counts.dtype))

        # Reshape back to [n_codes, book_size, embedding_dim]
        embed_sum = embed_sum.view(self.n_codes, self.book_size, self.embedding_dim)
        counts = counts.view(self.n_codes, self.book_size)

        # Compute new centers
        mask = counts > 0
        new_centers = codebook.clone()
        new_centers[mask] = embed_sum[mask] / counts[mask].unsqueeze(1)

        # EMA update: only update codes that were used
        codebook[mask] = self.momentum * codebook[mask] + (1 - self.momentum) * new_centers[mask]
        return
