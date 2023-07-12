from __future__ import annotations
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass

from ..walls import Wall


class Strain(metaclass=ABCMeta):
    def __init__(self, wall: Wall):
        self.wall = wall

    @abstractmethod
    def input(self, x: float, y: float, t: float) -> float:
        ...
