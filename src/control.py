"""A utility module providing various helper functions and classes for data processing and manipulation."""

from typing import Any, ClassVar


class Singleton(type):
    """A metaclass that ensures only one instance of a class is created."""

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


# Allows a temporary change using the "with" clause
class UsuallyFalse:
    """A class that provides a context manager for temporarily changing a boolean value."""

    _value: bool = False

    def __bool__(self) -> bool:
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False
