from typing import Any

import torch
from torch.autograd import Function

from structural_losses.structural_losses_backend import ApproxMatch, MatchCost, MatchCostGrad


# Inherit from Function
class MatchCostFunction(Function):
    # Note that both forward and backward are static methods
    @staticmethod
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        input:
            set1 : batch_size * #dataset_points * 3
            set2 : batch_size * #query_points * 3
        returns:
            match : batch_size * #query_points * #dataset_points
        """
        set1, set2, *_ = args
        # bias is an optional argument
        ctx.save_for_backward(set1, set2)

        match, temp = ApproxMatch(set1, set2)
        ctx.match = match
        cost = MatchCost(set1, set2, match)
        return cost

    # This function has only a single output, so it uses only one gradient
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_output = grad_outputs[0]
        set1, set2 = ctx.saved_tensors
        grad1, grad2 = MatchCostGrad(set1, set2, ctx.match)
        grad_output_expand = grad_output.unsqueeze(1).unsqueeze(2)
        return grad1 * grad_output_expand, grad2 * grad_output_expand


match_cost = MatchCostFunction.apply
