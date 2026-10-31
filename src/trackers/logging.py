"""Built-in logger handling the events of this project."""

import functools

from typing import override

from torch import Tensor

from drytorch.core import log_events
from drytorch.trackers.logging import INFO_LEVELS, BuiltinLogger, logger

from src.lib.events import ConfusionMatrixEvent

__all__ = ['ClassificationLogger']


class ClassificationLogger(BuiltinLogger):
    """Built-in logger that also prints confusion matrices."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ConfusionMatrixEvent) -> None:
        args = {'source_name': event.source_name, 'table': self._format(event.matrix)}
        logger.log(INFO_LEVELS.test, '%(source_name)s:\n%(table)s', args)
        return super().notify(event)

    @staticmethod
    def _format(matrix: Tensor) -> str:
        """Return the counts as a table, labels on the rows."""
        counts: list[list[int]] = matrix.round().long().tolist()
        names = [str(i) for i in range(len(counts))]
        largest = str(max(max(row) for row in counts))
        width = max(len(name) for name in [*names, largest]) + 2
        lines = ['label \\ predicted', ' ' * width + ''.join(name.rjust(width) for name in names)]
        for name, row in zip(names, counts, strict=True):
            lines.append(name.rjust(width) + ''.join(str(count).rjust(width) for count in row))

        return '\n'.join(lines)
