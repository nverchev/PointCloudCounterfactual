import abc
import functools
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from sympy.physics.units import temperature

from src.data_structures import Inputs, Outputs, W_Inputs
from src.encoders import get_encoder, get_w_encoder, WEncoderConvolution, BaseWEncoder
from src.decoders import get_decoder, get_w_decoder
from src.neighbour_ops import pykeops_square_distance
from src.layers import TransferGrad
from src.utils import UsuallyFalse
from src.config_options import MainExperiment, ExperimentAE, ModelHead


class WAutoEncoder(nn.Module):

    def __init__(self, codebook: torch.Tensor) -> None:

        super().__init__()
        cfg = MainExperiment.get_config()
        cfg_ae = cfg.autoencoder
        cfg_model = cfg_ae.model
        self.num_classes = cfg.data.dataset.n_classes
        self.z_dim = cfg_model.z_dim
        self.codebook = codebook
        self.dim_codes, self.book_size, self.embedding_dim = codebook.data.size()
        self.n_pseudo_input = cfg_model.n_pseudo_inputs
        self.encoder: BaseWEncoder = WEncoderConvolution()
        self.decoder = get_w_decoder()
        dim_codes = cfg_model.w_dim // cfg_model.embedding_dim
        self.init_pseudo = nn.init.normal_
        self.pseudo_inputs = nn.Parameter(torch.empty(self.n_pseudo_input, cfg_model.embedding_dim, dim_codes))
        self.init_pseudo(self.pseudo_inputs)
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_input, cfg_model.z_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_input, cfg_model.z_dim))
        self.updated: bool = False

    def update_pseudo_latent(self) -> None:
        pseudo_data = self.encode(None)
        self.pseudo_mu = nn.Parameter(pseudo_data.pseudo_mu)
        self.pseudo_log_var = nn.Parameter(pseudo_data.pseudo_log_var)

    def forward(self, x: W_Inputs) -> Outputs:
        data = self.encode(x.w_q)
        return self.decode(data)

    def gaussian_sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encode(self, opt_x: Optional[torch.Tensor]) -> Outputs:
        if opt_x is None:
            x: torch.Tensor = self.pseudo_inputs
        else:
            x = opt_x.view(-1, self.dim_codes, self.embedding_dim).transpose(2, 1)
            x = torch.cat((x, self.pseudo_inputs), dim=0)
        data = Outputs()
        x = self.encoder(x)
        batch = x[:-self.n_pseudo_input or None]
        data.mu, data.log_var = batch.chunk(2, 1)
        data.p_mu, data.p_log_var = torch.zeros_like(data.mu), torch.zeros_like(data.log_var)
        if self.n_pseudo_input:
            data.pseudo_mu, data.pseudo_log_var = x[-self.n_pseudo_input:].chunk(2, 1)

        data.z = self.gaussian_sample(data.mu, data.log_var)
        return data

    def decode(self, data: Outputs) -> Outputs:
        data.w_recon = self.decoder(data.z)
        data.w_dist, data.idx = self.dist(data.w_recon)
        return data

    def recursive_reset_parameters(self) -> None:
        self.init_pseudo(self.pseudo_inputs)
        self.apply(lambda x: x.reset_parameters() if (hasattr(x, 'reset_parameters') or x is self.codebook) else x)

    def dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _ = x.shape
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.detach().repeat(batch, 1, 1)
        dist = pykeops_square_distance(x, book)
        # Lazy vector need aggregation like sum(1) to yield tensor (|dim 1| = 1)
        idx = dist.argmin(axis=2)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), idx.view(batch, self.dim_codes, 1)


class CounterfactualWAutoEncoder(WAutoEncoder):

    def __init__(self, codebook: torch.Tensor) -> None:
        super().__init__(codebook)
        cfg = MainExperiment.get_config()
        cfg_wae = cfg.autoencoder.model.encoder.w_encoder
        self.temperature = torch.tensor(cfg_wae.cf_temperature, dtype=torch.float)
        self.gumbel = cfg_wae.gumbel
        self.softmax = torch.nn.Softmax(dim=1)
        self.gumbel_softmax = functools.partial(nn.functional.gumbel_softmax, tau=self.temperature, hard=False)
        self.relaxed_softmax = lambda x: self.softmax(x / self.temperature)
        self.encoder = get_w_encoder()

    def encode(self, opt_x: torch.Tensor | None) -> Outputs:
        if opt_x is None:
            x: torch.Tensor = self.pseudo_inputs
        else:
            x = opt_x.view(-1, self.dim_codes, self.embedding_dim).transpose(2, 1)
            x = torch.cat((x, self.pseudo_inputs), dim=0)
        data = Outputs()
        encoded, prediction = self.encoder(x)
        data.y = self.softmax(prediction[:-self.n_pseudo_input or None])
        data.mu, data.log_var = encoded[:-self.n_pseudo_input or None].chunk(2, 1)
        return data

    def decode(self, data: Outputs, logits: Optional[torch.Tensor] = None) -> Outputs:
        if logits is not None:
            data.z_c = self.gumbel_softmax(logits) if self.gumbel and self.training else self.relaxed_softmax(logits)
        if self.n_pseudo_input:
            data.pseudo_mu, data.pseudo_log_var = data.h[-self.n_pseudo_input:].chunk(2, 1)
        data.z = self.gaussian_sample(data.mu, data.log_var)
        data.w_recon = self.decoder(torch.cat((data.z, data.z_c), dim=1))
        data.w_dist, data.idx = self.dist(data.w_recon)
        return data

    def forward(self, x: W_Inputs) -> Outputs:
        data = self.encode(x.w_q)
        return self.decode(data, x.logits)


class AutoEncoder(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()
        cfg = MainExperiment.get_config()
        cfg_ae = cfg.autoencoder
        self.m_training = cfg_ae.model.training_output_points
        self.m_test = cfg_ae.objective.n_inference_output_points

    @property
    def m(self) -> int:
        return self.m_test if torch.is_inference_mode_enabled() else self.m_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        ...

    def recursive_reset_parameters(self) -> None:
        self.apply(lambda x: x.reset_parameters() if hasattr(x, 'reset_parameters') else x)


class Oracle(AutoEncoder):

    def forward(self, inputs: Inputs) -> Outputs:
        data = Outputs()
        data.recon = inputs.cloud[:, :self.m, :]
        return data


class AE(AutoEncoder):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()

    def forward(self, inputs: Inputs) -> Outputs:
        data = self.encode(inputs.cloud, inputs.indices)
        return self.decode(data, inputs.initial_sampling, inputs.viz_att, inputs.viz_components)

    def encode(self, x: torch.Tensor, indices: torch.Tensor) -> Outputs:
        data = Outputs()
        data.w = self.encoder(x, indices)
        return data

    def decode(self,
               data: Outputs,
               initial_sampling: torch.Tensor = torch.empty(0),
               viz_att: torch.Tensor = torch.empty(0),
               viz_components: torch.Tensor = torch.empty(0)) -> Outputs:
        x = self.decoder(data.w, self.m, initial_sampling, viz_att, viz_components)
        data.recon = x.transpose(2, 1).contiguous()
        return data


class VQVAE(AE):

    def __init__(self) -> None:
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__()
        cfg_ae = ExperimentAE.get_config().model
        self.double_encoding = UsuallyFalse()
        self.num_codes = cfg_ae.n_codes
        self.book_size = cfg_ae.book_size
        self.embedding_dim = cfg_ae.embedding_dim
        self.vq_ema_update = cfg_ae.vq_ema_update
        self.codebook = torch.nn.Parameter(
            torch.randn(self.num_codes, self.book_size, self.embedding_dim, requires_grad=not cfg_ae.vq_ema_update)
        )
        if cfg_ae.vq_ema_update:
            self.decay = .999
            self.gain = 1 - self.decay
            self.ema_counts = torch.nn.Parameter(torch.ones(self.num_codes, self.book_size, dtype=torch.float))

        self.w_autoencoder = CounterfactualWAutoEncoder(self.codebook)
        self.transfer = TransferGrad()

    def quantise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, embed = x.size()
        x = x.view(batch * self.num_codes, 1, self.embedding_dim)
        book = self.codebook.repeat(batch, 1, 1)
        dist = pykeops_square_distance(x, book)
        idx = dist.argmin(axis=2).view(-1, 1, 1)
        cw_embed = self.get_quantised_code(idx, book)
        one_hot_idx = torch.zeros(batch, self.num_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.num_codes, 1), 1)
        # EMA update
        if self.training and self.vq_ema_update:
            x = x.view(batch, self.num_codes, self.embedding_dim).transpose(0, 1)
            idx = idx.view(batch, self.num_codes, 1).transpose(0, 1).expand(-1, -1, self.embedding_dim)
            update_dict = torch.zeros_like(self.codebook).scatter_(index=idx, src=x, dim=1, reduce='sum')
            normalize = self.ema_counts.unsqueeze(2).expand(-1, -1, self.embedding_dim)
            self.codebook.data = self.codebook * self.decay + self.gain * update_dict / (normalize + 1e-6)
            self.ema_counts.data = self.decay * self.ema_counts + self.gain * one_hot_idx.sum(0)

        return cw_embed, one_hot_idx

    def encode(self, x: torch.Tensor, indices: torch.Tensor) -> Outputs:
        data = Outputs()
        data.w_q = self.encoder(x, indices)
        if self.double_encoding:
            data.update(self.w_autoencoder.encode(data.w_q.detach()))
        return data

    def decode(self,
               data: Outputs,
               initial_sampling: torch.Tensor = torch.empty(0),
               viz_att: torch.Tensor = torch.empty(0),
               viz_components: torch.Tensor = torch.empty(0)) -> Outputs:
        if self.double_encoding:
            self.w_autoencoder.decode(data)  # looks for the z keyword
            idx = data.idx
            batch = idx.shape[0]
            book = self.codebook.repeat(batch, 1, 1)
            data.w_e = data.w = self.get_quantised_code(idx.view(batch * idx.shape[1], 1, 1), book)
        else:
            data.w_e, data.one_hot_idx = self.quantise(data.w_q)
            data.w = self.transfer.apply(data.w_e, data.w_q)
        return super().decode(data, initial_sampling, viz_att, viz_components)

    @torch.inference_mode()
    def random_sampling(self,
                        batch_size: int,
                        initial_sampling: torch.Tensor = torch.empty(0),
                        viz_att: torch.Tensor = torch.empty(0),
                        viz_components: torch.Tensor = torch.empty(0),
                        z_bias: int = 0) -> Outputs:
        self.eval()
        pseudo_mu = self.w_autoencoder.pseudo_mu
        pseudo_log_var = self.w_autoencoder.pseudo_log_var
        pseudo_z_list: list[torch.Tensor] = []
        for _ in range(batch_size):
            i = np.random.randint(self.w_autoencoder.n_pseudo_input)
            pseudo_z_list.append(self.w_autoencoder.gaussian_sample(pseudo_mu[i], pseudo_log_var[i]))
        pseudo_z = torch.stack(pseudo_z_list).contiguous()
        data = Outputs()
        data.z = pseudo_z + z_bias
        with self.double_encoding:
            out = self.decode(data, initial_sampling, viz_att, viz_components)
        return out

    def get_quantised_code(self, idx: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        idx = idx.expand(-1, -1, self.embedding_dim)
        return book.gather(1, idx).view(-1, self.num_codes * self.embedding_dim)


def get_module() -> AutoEncoder:
    model_map: dict[ModelHead, Type[AutoEncoder]] = {ModelHead.AE: AE, ModelHead.VQVAE: VQVAE}
    return model_map[ExperimentAE.get_config().model.head]()
