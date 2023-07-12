from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..walls import Wall


@dataclass(frozen=True)
class Obstacle:
    """_summary_
        障害物を表現するクラス. Wallのファーストクラスコレクション.
    Args:
        walls: Wallの配列.
    """

    walls: list[Wall]

    def __post_init__(self):
        if len(self.walls) < 2:
            raise ValueError("障害物は2つ以上必要.")
        if not self._are_all_vertical_intersecting:
            raise ValueError("全て垂直に交わるように配置する必要がある．")

    def _are_all_vertical_intersecting(self):
        for wall, next_wall in zip(self.walls, self.walls[1:]):
            vec_1 = np.array(wall.pt1) - np.array(wall.pt2)
            vec_2 = np.array(next_wall.pt1) - np.array(next_wall.pt2)
            if np.dot(vec_1, vec_2):
                return False
        return True

    def xs(self):
        return np.array([wall.xs() for wall in self.walls]).ravel().tolist()

    def ys(self):
        return np.array([wall.ys() for wall in self.walls]).ravel().tolist()

    def locations(self):
        return [wall.location for wall in self.walls]
