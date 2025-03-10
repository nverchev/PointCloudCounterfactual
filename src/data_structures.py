import dataclasses
from typing import NamedTuple, Self

import torch

IN_CHAN = 3
OUT_CHAN = 3


class Inputs(NamedTuple):
    cloud: torch.Tensor
    indices: torch.Tensor = torch.empty(0)
    initial_sampling: torch.Tensor = torch.empty(0)
    viz_att: torch.Tensor = torch.empty(0)
    viz_components: torch.Tensor = torch.empty(0)


class Targets(NamedTuple):
    ref_cloud: torch.Tensor
    scale: torch.Tensor = torch.FloatTensor([1.0])
    label: torch.Tensor = torch.tensor(0)


@dataclasses.dataclass(init=False, slots=True)
class Outputs:
    recon: torch.Tensor
    w: torch.Tensor
    w_q: torch.Tensor
    w_e: torch.Tensor
    w_recon: torch.Tensor
    w_dist: torch.Tensor
    idx: torch.Tensor
    one_hot_idx: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    p_mu: torch.Tensor
    p_log_var: torch.Tensor
    pseudo_mu: torch.Tensor
    pseudo_log_var: torch.Tensor
    z: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor
    z_c: torch.Tensor

    def update(self, other: Self) -> None:
        for attribute in other.__slots__:
            try:
                setattr(self, attribute, getattr(other, attribute))
            except AttributeError:  # slot not yet initialized
                pass


class W_Targets(NamedTuple):
    one_hot_idx: torch.Tensor
    logits: torch.Tensor


class W_Inputs(NamedTuple):
    w_q: torch.Tensor
    logits: torch.Tensor
