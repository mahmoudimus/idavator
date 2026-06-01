"""Event emitter.

Vendored from  (``/core/events.py``) with its typing imports swapped for
the standard library. A tiny generic pub/sub: ``emit()`` calls handlers inline, so
any asynchronous behaviour lives in what a handler does (e.g. enqueue), not here.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

E = TypeVar("E", bound=Hashable)


@dataclasses.dataclass
class EventEmitter(Generic[E]):
    _listeners: collections.defaultdict[E, set[Callable]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set), init=False
    )

    def on(self, event: E, handler: Callable | None = None):
        """Register an event handler for the given event."""
        if handler:
            self._listeners[event].add(handler)
            return handler

        @functools.wraps(self.on)
        def decorator(func):
            self.on(event, func)
            return func

        return decorator

    def once(self, event: E, handler: Callable):
        @functools.wraps(handler)
        def once_handler(*args, **kwargs):
            self.remove(event, once_handler)
            return handler(*args, **kwargs)

        self.on(event, once_handler)

    def remove(self, event: E, handler: Callable):
        self._listeners[event].discard(handler)

    def clear(self):
        self._listeners.clear()

    def emit(self, event: E, *args, **kwargs):
        for handler in list(self._listeners[event]):
            handler(*args, **kwargs)
