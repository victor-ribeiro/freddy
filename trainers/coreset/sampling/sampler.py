from functools import singledispatchmethod
from collections.abc import Callable
from freddy.lazzy_greed import freddy

METHOD = {"coreset": freddy, "random": None}


class Sampler:

    def __init__(self, method: str = "random") -> None:
        self.method = METHOD[method]

    def _(self, method: Callable):
        pass
