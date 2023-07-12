from __future__ import annotations
from enum import Enum


class ReflectCondition(Enum):
    NEUMANN = 0
    DIRICRE = 1


class Location(Enum):
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3
    RIGHTTOP = 4
    LEFTTOP = 5
    RIGHTBOTTOM = 6
    LEFTBOTTOM = 7


XIndices = list[int]
YIndices = list[int]
XYIndices = tuple[XIndices, YIndices]
