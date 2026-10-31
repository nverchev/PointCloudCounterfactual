"""Outer autoencoder: the VQ-VAE turning point clouds into discrete codes."""

import abc

from typing import Final, override

import torch
import torch.nn as nn

from pykeops.torch import LazyTensor

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs, Quantization
from src.module.decoders import PCGen
from src.module.encoders import DGCNNEncoder
from src.module.layers import TransferGrad, Transpose
from src.utils.neighbour_ops import pykeops_square_distance

__all__ = ['VQVAE']


class BaseAutoencoder(nn.Module, abc.ABC):
    """Abstract base of the outer autoencoders.

    Attributes:
        dim_codes: Number of codes of a codeword.
        book_size: Codes of each codebook.
        embedding_dim: Dimension of each code.
        n_points: Points of the clouds it decodes.
    """

    dim_codes: Final[int]
    book_size: Final[int]
    embedding_dim: Final[int]
    n_points: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_model = cfg.autoencoding.model
        self.dim_codes = cfg_model.dim_codes
        self.book_size = cfg_model.book_size
        self.embedding_dim = cfg_model.embedding_dim
        self.n_points = cfg.data.n_points
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Encode the clouds and decode them."""


class VQVAE(BaseAutoencoder):
    """Vector-quantized point cloud autoencoder.

    Attributes:
        dim_codes: Chunks of a codeword, one codebook each.
        book_size: Codes of each codebook.
        embedding_dim: Dimension of each code.
        n_points: Points of the clouds it decodes.
        encoder: Point encoder.
        decoder: Point decoder.
        transpose: Moves the coordinates of the decoded clouds to the last dimension.
        codebook: Codes ``(dim_codes, book_size, embedding_dim)``.
    """

    encoder: DGCNNEncoder
    decoder: PCGen
    transpose: Transpose
    codebook: nn.Parameter

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.encoder = DGCNNEncoder()
        self.decoder = PCGen()
        self.transpose = Transpose(2, 1)
        self.codebook = nn.Parameter(torch.randn(self.dim_codes, self.book_size, self.embedding_dim))
        return

    @override
    def forward(self, inputs: Inputs) -> Outputs:
        """Encode the clouds, quantize the codewords and decode them."""
        w_q = self.encode(inputs.cloud)
        w_e, one_hot_idx = self.quantise(w_q)
        recon = self._decode(TransferGrad.apply(w_e, w_q), inputs.initial_sampling)
        return Outputs(recon, Quantization(w_q, w_e, one_hot_idx))

    def decode_nearest_codes(self, w_recon: torch.Tensor, /, initial_sampling: torch.Tensor) -> Outputs:
        """Decode the codes nearest to a reconstructed codeword.

        Args:
            w_recon: codewords ``(B, W)``.
            initial_sampling: sampling of the generating surface ``(B, D, M)``; empty to draw one.

        Returns:
            The decoded clouds.
        """
        w_e, _ = self.quantise(w_recon)
        return Outputs(self._decode(w_e, initial_sampling))

    def encode(self, x: torch.Tensor, /) -> torch.Tensor:
        """Encode the clouds ``(B, N, IN_CHAN)`` into the codewords before quantization ``(B, W)``."""
        return self.encoder(x)

    def quantise(self, x: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace each chunk of the codewords with the nearest code of its codebook.

        Args:
            x: codewords ``(B, W)``.

        Returns:
            The quantized codewords ``(B, W)`` and the indicator of the codes ``(B, dim_codes, book_size)``.
        """
        batch, _ = x.size()
        book = self.codebook.repeat(batch, 1, 1)
        idx = self._lazy_square_distances(x, book).argmin(axis=2).view(-1, 1, 1)
        idx_expanded = idx.expand(-1, -1, self.embedding_dim)
        w_e = book.gather(1, idx_expanded).view(-1, self.dim_codes * self.embedding_dim)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        return w_e, one_hot_idx

    def square_distances(self, x: torch.Tensor, /) -> torch.Tensor:
        """Squared distance of every chunk of a codeword from every code of its group.

        The codebook is detached: the distances choose codes, and do not train them.

        Args:
            x: codewords ``(B, W)``.

        Returns:
            The distances ``(B, dim_codes, book_size)``.
        """
        batch, _ = x.size()
        book = self.codebook.detach().repeat(batch, 1, 1)
        return self._lazy_square_distances(x, book).sum(1).view(batch, self.dim_codes, self.book_size)

    def _decode(self, w: torch.Tensor, sampling: torch.Tensor) -> torch.Tensor:
        return self.transpose(self.decoder(w, self.n_points, sampling.to(w.device))).contiguous()

    def _lazy_square_distances(self, x: torch.Tensor, book: torch.Tensor) -> LazyTensor:
        batch, _ = x.size()
        chunks = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        return pykeops_square_distance(chunks, book)
