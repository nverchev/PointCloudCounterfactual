"""Module where the parallelism is defined: change according to your needs."""
import logging
import os
from typing import Callable, Generic, ParamSpec, TypeVar
from multiprocessing import managers
import torch.multiprocessing as mp
import socket

import torch
import torch.distributed as dist

P = ParamSpec('P')
T = TypeVar('T')

class DistributedWorker(Generic[P, T]):
    """Callable wrapper to distribute a worker across multiple processes."""

    def __init__(self, worker: Callable[P, T], world_size: int) -> None:
        """Initialize."""
        self.worker = worker
        self.world_size = world_size
        self.port = self._get_free_port()
        return

    def __call__(self, rank: int, *args: P.args) -> None:
        """Run the worker in a distributed environment."""
        self._setup_distributed(rank)
        try:
            return self.worker(*args)
        finally:
            self._cleanup_distributed()

    def spawn(self, *args: P.args) -> None:
        """Spawn the worker in multiple processes."""
        return mp.spawn(self, args=args, nprocs=self.world_size)

    def _setup_distributed(self, rank: int) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(backend,
                                rank=rank,
                                init_method=f'tcp://127.0.0.1:{self.port}',
                                world_size=self.world_size)

        return

    @staticmethod
    def _get_free_port() -> str:
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return str(s.getsockname()[1])

    @staticmethod
    def _cleanup_distributed() -> None:
        """Clean up the distributed environment."""
        dist.destroy_process_group()
        return
