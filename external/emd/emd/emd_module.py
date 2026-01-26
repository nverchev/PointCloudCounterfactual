"""EMD approximation module (based on auction algorithm)"""

# memory complexity: O(n)
# time complexity: O(n^2 * iter)
# author: Minghua Liu

from typing import Any, cast, override

from emd import emd_backend
import torch

from torch import nn
from torch.autograd import Function


class emdFunction(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        xyz1, xyz2, eps, iters, *_ = args
        batch_size1, n, _ = xyz1.size()
        batch_size2, m, _ = xyz2.size()

        if n != m:
            raise ValueError('Input point clouds should have the same number of points')
        if batch_size1 != batch_size2:
            raise ValueError('Batch size must be the same')
        if n % 1024:
            raise ValueError('Only valid for clouds of a size multiple of 1024')
        if batch_size1 > 512:
            raise ValueError('Batch size should not exceed 512')

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batch_size1, n, device='cuda').contiguous()
        assignment = torch.zeros(batch_size1, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batch_size1, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batch_size1, m, device='cuda').contiguous()
        bid = torch.zeros(batch_size1, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batch_size1, n, device='cuda').contiguous()
        max_increments = torch.zeros(batch_size1, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batch_size1 * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batch_size1 * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd_backend.forward(
            xyz1,
            xyz2,
            dist,
            assignment,
            price,
            assignment_inv,
            bid,
            bid_increments,
            max_increments,
            unass_idx,
            unass_cnt,
            unass_cnt_sum,
            cnt_tmp,
            max_idx,
            eps,
            iters,
        )

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_dist, *_ = grad_outputs
        xyz1, xyz2, assignment = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()

        grad_xyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        grad_xyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd_backend.backward(xyz1, xyz2, grad_xyz1, grad_dist, assignment)
        return grad_xyz1, grad_xyz2, None, None


class emdModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, input1: torch.Tensor, input2: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
        """Compute the earth mover's distance (EMD) between two point clouds normalized to [0, 1] and of the same size.

        Args:
            input1: predicted point cloud [#batch, #points, 3] | #batch <= 512 an #points is a multiple of 1024
            input2: ground truth point cloud [#batch, #points, 3] | #batch <= 512 an #points is a multiple of 1024
            eps: a parameter which balances the error rate and the speed of convergence
            iters: the number of iteration

        Returns:
            dist: [#batch, #points] |  sqrt(dist) -> L2 distance
            assignment: [#batch, #points] index of the matched point in the ground truth point cloud
        """
        return cast(torch.Tensor, emdFunction.apply(input1, input2, eps, iters))
