"""Module containing the Flow Matching models."""

import abc

import torch
import torch.nn as nn
import itertools

from src.config import Experiment
from src.config.specs import FlowExperimentConfig
from src.data.structures import Inputs, Outputs
from src.module.autoencoders import BaseVAE
from src.module.decoders import get_decoder, BasePointDecoder
from src.module.time import get_time_embedding
from src.module.latent_decoders import get_latent_decoder
from src.config.options import FlowModels, TimeSamplers
from src.utils.neighbour_ops import cluster_wise_assign, get_bijective_assignment


class BaseFlow(nn.Module, abc.ABC):
    """Abstract Base Class for Flow Matching Models."""

    decoder: BasePointDecoder
    n_training_points_training: int
    n_inference_output_points: int
    time_embedding_dim: int
    mlp_dims: tuple[int, ...]
    d_k: int
    noise_scale: torch.Tensor

    def __init__(self, cfg_flow: FlowExperimentConfig):
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoder
        cfg_model = cfg_flow.model
        decoder_cfg = cfg_model.decoder if cfg_model.decoder is not None else cfg_ae.model.decoder

        self._setup_stage()
        self.assign_noise: bool = cfg_model.assign_noise
        self.s_k: float = cfg_flow.s_k
        self.register_buffer('noise_scale', torch.tensor(1.0 / self.d_k**0.5))
        self.decoder = get_decoder(decoder_cfg, cfg_model.feature_dim)
        self.time_sampling: TimeSamplers = cfg_flow.objective.time_sampling
        return

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def n_output_points(self) -> int:
        """Get the number of output points."""
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training velocity prediction."""

    @abc.abstractmethod
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_timesteps: int,
        n_points: int,
        device: torch.device,
        x_prev: torch.Tensor | None = None,
        z1: torch.Tensor | None = None,
        z2: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Sample from the flow matching model."""

    @abc.abstractmethod
    def _setup_stage(self) -> None:
        """Setup the stage."""

    @abc.abstractmethod
    def _order_noise(self, x_1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Order noise to match x_1."""

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample global time t in [0.0, 1.0] according to the sampling strategy."""
        u = torch.rand(batch_size, device=device).view(-1, 1, 1)
        if self.time_sampling == TimeSamplers.SquareRoot:
            return torch.sqrt(u)

        return u

    def _sample_noise(self, shape: torch.Size | tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample n ~ N(0, 1/D^k * I) as per Eq. 6."""
        return torch.randn(shape, device=device) * self.noise_scale

    def _sample_alignment_noise(self, n_samples: int, n_points: int, device: torch.device) -> torch.Tensor:
        """Sample n' ~ N(0, Sigma') as per Eq. 24/70."""
        # Sigma' = (1-s_k)^2 / D^k * I  (assuming e_k always equals 1)
        std = (1.0 - self.s_k) * self.noise_scale
        return torch.randn(n_samples, n_points, 3, device=device) * std


class FlowMatching(BaseFlow, abc.ABC):
    """Core Flow Matching logic for Point Clouds (Stages 1, 2, 3)."""

    def __init__(self, cfg_flow: FlowExperimentConfig):
        super().__init__(cfg_flow)
        self.time_embedding = get_time_embedding(
            cfg=cfg_flow.model.time_embedding,
            feature_dim=cfg_flow.model.feature_dim,
        )
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training velocity prediction (MFM Algorithm 1)."""
        batch_size, _, _ = inputs.cloud.shape
        device = inputs.cloud.device
        x_1_clean = self._get_target(inputs)
        x_0_clean = self._get_source(inputs)
        features = self._get_features(inputs)
        t = self._sample_time(batch_size, device=device)
        emb_features = self.time_embedding(t.view(batch_size, 1), features)
        n = self._sample_noise(x_1_clean.shape, device)
        x_s = self.s_k * x_0_clean + (1.0 - self.s_k) * n
        if self.assign_noise:
            x_s = self._order_noise(x_1_clean, x_s)

        x_t = (1.0 - t) * x_s + t * x_1_clean
        v_pred = self.decoder(x_t, emb_features, self.n_output_points)
        out = Outputs()
        out.v_pred = v_pred
        out.v_target = x_1_clean - x_s
        return out

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_timesteps: int,
        n_points: int,
        device: torch.device,
        x_prev: torch.Tensor | None = None,
        z1: torch.Tensor | None = None,
        z2: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Deterministic ODE Sampling (Euler Integration)."""
        if x_prev is None:
            x_t = self._sample_noise((n_samples, n_points, 3), device)
            x_0_clean = torch.zeros_like(x_t)
        else:
            if x_prev.shape[1] < n_points:
                ratio = n_points // x_prev.shape[1]
                x_0_clean = x_prev.repeat_interleave(ratio, dim=1)
            else:
                x_0_clean = x_prev

            x_t = x_0_clean * self.s_k + self._sample_alignment_noise(n_samples, n_points, device)

        x_list = [x_t]
        features = self._decode_latent(z1, z2, n_samples, device)
        for i in range(n_timesteps):
            t_curr = i / n_timesteps
            t_next = (i + 1) / n_timesteps
            dt = t_next - t_curr
            t_tensor = torch.full((n_samples,), t_curr, device=device).view(-1, 1)
            emb_features = self.time_embedding(t_tensor, features)
            v_pred = self.decoder(x_t, emb_features, n_points)
            x_t = x_t + v_pred * dt
            if i % 10 == 0:
                x_list.append(x_t)

        x_list.append(x_t)
        return x_list

    def _decode_latent(
        self, z1: torch.Tensor | None, z2: torch.Tensor | None, n_samples: int, device: torch.device
    ) -> torch.Tensor:
        """Decode latent variables to features (default zeros)."""
        return torch.zeros(n_samples, self.decoder.feature_dim, device=device)

    def _get_features(self, inputs: Inputs) -> torch.Tensor:
        """Get zero features."""
        return self._decode_latent(None, None, inputs.cloud.shape[0], inputs.cloud.device)

    @abc.abstractmethod
    def _get_target(self, inputs: Inputs) -> torch.Tensor:
        """Get target point cloud (X^k_1)."""

    @abc.abstractmethod
    def _get_source(self, inputs: Inputs) -> torch.Tensor:
        """Get source point cloud (Up(X^{k+1}_1))."""


class CondFlowMatching(FlowMatching):
    """Flow Matching Model that integrates with a pre-trained autoencoder."""

    def __init__(self, autoencoder: BaseVAE, cfg_flow: FlowExperimentConfig):
        super().__init__(cfg_flow)
        self.autoencoder: BaseVAE = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.autoencoder.eval()
        self.latent_decoder = get_latent_decoder(
            cfg_flow.model.latent_decoder,
            z1_dim=autoencoder.z1_dim,
            z2_dim=autoencoder.z2_dim,
            feature_dim=cfg_flow.model.feature_dim,
        )
        return

    def _decode_latent(
        self, z1: torch.Tensor | None, z2: torch.Tensor | None, n_samples: int, device: torch.device
    ) -> torch.Tensor:
        """Decode latent variables to features using the autoencoder's latent decoder."""
        if z1 is None or z2 is None:
            ae_out = self.autoencoder.generate(batch_size=n_samples)
            z1 = ae_out.z1 if z1 is None else z1
            z2 = ae_out.z2 if z2 is None else z2

        return self.latent_decoder(z1, z2)

    def _get_features(self, inputs: Inputs) -> torch.Tensor:
        """Get features from the autoencoder."""
        with torch.no_grad():
            out = self.autoencoder.encode(inputs)

        return self._decode_latent(out.z1, out.z2, inputs.cloud.shape[0], inputs.cloud.device)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_timesteps: int,
        n_points: int,
        device: torch.device,
        x_prev: torch.Tensor | None = None,
        z1: torch.Tensor | None = None,
        z2: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Sample from the flow, using provided latents or sampling from prior."""
        return super().sample(
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            n_points=n_points,
            device=device,
            x_prev=x_prev,
            z1=z1,
            z2=z2,
        )


class ClusterPermutation:
    """Permutation utility class."""

    def __init__(self):
        self.perms = self._precompute_permutations(4)
        return

    def cluster_wise_assign(self, x: torch.Tensor, noise: torch.Tensor, k: int) -> torch.Tensor:
        return cluster_wise_assign(x, noise, self.perms, k)

    def _precompute_permutations(self, k: int) -> torch.Tensor:
        """Precompute all permutations of length k."""
        return torch.tensor(list(itertools.permutations(range(k))))


class Stage1Mixin:
    """Configuration mixin for Flow Stage 1."""

    def _setup_stage(self) -> None:
        """Setup stage 1."""
        cfg = Experiment.get_config()
        self.n_training_points_training = cfg.data.n_input_points
        self.n_inference_output_points = cfg.data.n_target_points
        self.cluster_permutation = ClusterPermutation()
        self.d_k = 1
        return

    def _order_noise(self, x_1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Order noise to match x_1 cluster-wise (512 clusters of 4)."""
        return self.cluster_permutation.cluster_wise_assign(x_1, noise, k=4)

    def _get_target(self, inputs: Inputs) -> torch.Tensor:
        return inputs.cloud

    def _get_source(self, inputs: Inputs) -> torch.Tensor:
        return inputs.cloud_512.repeat_interleave(4, dim=1)


class Stage2Mixin:
    """Configuration mixin for Flow Stage 2."""

    def _setup_stage(self) -> None:
        """Setup stage 2."""
        self.n_training_points_training = 512
        self.n_inference_output_points = 512
        self.cluster_permutation = ClusterPermutation()
        self.d_k = 4
        return

    def _order_noise(self, x_1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Order noise to match x_1 cluster-wise (128 clusters of 4)."""
        return self.cluster_permutation.cluster_wise_assign(x_1, noise, k=4)

    def _get_target(self, inputs: Inputs) -> torch.Tensor:
        return inputs.cloud_512

    def _get_source(self, inputs: Inputs) -> torch.Tensor:
        return inputs.cloud_128.repeat_interleave(4, dim=1)


class Stage3Mixin:
    """Configuration mixin for Flow Stage 3."""

    def _setup_stage(self) -> None:
        """Setup stage 3."""
        self.n_training_points_training = 128
        self.n_inference_output_points = 128
        self.d_k = 16
        return

    def _get_target(self, inputs: Inputs) -> torch.Tensor:
        return inputs.cloud_128

    def _get_source(self, inputs: Inputs) -> torch.Tensor:
        return torch.zeros_like(inputs.cloud_128)

    def _order_noise(self, x_1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Order noise to match x_1 cluster-wise (16 clusters of 16)."""
        batch_size, n_points = x_1.shape[:2]
        device = x_1.device
        indices = get_bijective_assignment(x_1, noise)
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size, n_points)
        noise_aligned = noise[batch_idx, indices]
        return noise_aligned


class FlowStage1(Stage1Mixin, FlowMatching):
    """Flow Matching Stage 1: 512 to full size."""

    def __init__(self, cfg_flow: FlowExperimentConfig):
        super().__init__(cfg_flow)
        return


class FlowStage2(Stage2Mixin, FlowMatching):
    """Flow Matching Stage 2: 128 to 512 points."""

    def __init__(self, cfg_flow: FlowExperimentConfig):
        super().__init__(cfg_flow)
        return


class FlowStage3(Stage3Mixin, FlowMatching):
    """Flow Matching Stage 3: Noise to 128 points."""

    def __init__(self, cfg_flow: FlowExperimentConfig):
        super().__init__(cfg_flow)
        return


class CondFlowStage1(Stage1Mixin, CondFlowMatching):
    """Conditional Flow Matching Stage 1: 512 to full size."""

    def __init__(self, cfg_flow: FlowExperimentConfig, autoencoder: BaseVAE):
        super().__init__(autoencoder, cfg_flow)
        return


class CondFlowStage2(Stage2Mixin, CondFlowMatching):
    """Conditional Flow Matching Stage 2: 128 to 512 points."""

    def __init__(self, cfg_flow: FlowExperimentConfig, autoencoder: BaseVAE):
        super().__init__(autoencoder, cfg_flow)
        return


class CondFlowStage3(Stage3Mixin, CondFlowMatching):
    """Conditional Flow Matching Stage 3: Noise to 128 points."""

    def __init__(self, cfg_flow: FlowExperimentConfig, autoencoder: BaseVAE):
        super().__init__(autoencoder, cfg_flow)
        return


class FlowReconstruction(nn.Module):
    """Wrapper for multi-stage flow reconstruction evaluation."""

    def __init__(
        self,
        autoencoder: BaseVAE,
        stage1: CondFlowMatching,
        stage2: CondFlowMatching,
        stage3: CondFlowMatching,
    ):
        super().__init__()
        cfg = Experiment.get_config()
        self.autoencoder = autoencoder
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.n_timesteps_s1: int = cfg.flow_stage1.objective.n_timesteps
        self.n_timesteps_s2: int = cfg.flow_stage2.objective.n_timesteps
        self.n_timesteps_s3: int = cfg.flow_stage3.objective.n_timesteps
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass performing multi-stage reconstruction."""
        device = inputs.cloud.device
        batch_size = inputs.cloud.shape[0]
        n_final = inputs.cloud.shape[1]
        with torch.no_grad():
            ae_out = self.autoencoder.encode(inputs)
            z1 = ae_out.mu1
            z2 = ae_out.p_mu2 + ae_out.d_mu2

        stages = [
            (self.stage3, self.n_timesteps_s3, 128),
            (self.stage2, self.n_timesteps_s2, 512),
            (self.stage1, self.n_timesteps_s1, n_final),
        ]

        x_current: torch.Tensor | None = None
        for stage, n_timesteps, n_points in stages:
            x_current = stage.sample(
                n_samples=batch_size,
                n_timesteps=n_timesteps,
                n_points=n_points,
                device=device,
                x_prev=x_current,
                z1=z1,
                z2=z2,
            )[-1]

        assert x_current is not None
        out = Outputs()
        out.recon = x_current
        return out


def get_flow_module(cfg_flow: FlowExperimentConfig, autoencoder: BaseVAE | None = None) -> BaseFlow:
    """Get the correct flow matching module according to the configuration."""
    stage = cfg_flow.stage
    class_name = cfg_flow.model.class_name

    if class_name == FlowModels.FlowMatching:
        if stage == 1:
            return FlowStage1(cfg_flow)

        if stage == 2:
            return FlowStage2(cfg_flow)

        if stage == 3:
            return FlowStage3(cfg_flow)

        raise ValueError(f'Invalid stage {stage} for FlowMatching')

    if class_name == FlowModels.CondFlowMatching:
        if autoencoder is None:
            raise ValueError('Autoencoder must be provided for CondFlowMatching')

        if stage == 1:
            return CondFlowStage1(cfg_flow, autoencoder)

        if stage == 2:
            return CondFlowStage2(cfg_flow, autoencoder)

        if stage == 3:
            return CondFlowStage3(cfg_flow, autoencoder)

        raise ValueError(f'Invalid stage {stage} for CondFlowMatching')

    raise ValueError(f'Unknown class_name {class_name}')
