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
