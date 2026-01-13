"""Module for classes pertaining to control flow."""


class UsuallyFalse:
    """A class that provides a context manager for temporarily changing a boolean value."""

    _value: bool = False

    def __bool__(self) -> bool:
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False
