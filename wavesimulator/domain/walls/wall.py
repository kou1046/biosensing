from __future__ import annotations
from dataclasses import dataclass
from ..types import Location


@dataclass(frozen=True)
class Wall:
    """_summary_
        障害物やひずみの場所を表現するときに用いるクラス．
    Args:
        pt1: 始点
        pt2: 終点
        location: 壁の種類 (右 or 左 or 上 or 下)
    """

    pt1: tuple[float, float]
    pt2: tuple[float, float]
    location: Location

    def __post_init__(self):
        if self.is_upward() and self.is_horizontal():
            raise ValueError("locationの指定方法が間違っている")
        if self.is_downward() and self.is_horizontal():
            raise ValueError("locationの指定方法が間違っている")
        if self.is_rightward() and self.is_vertical():
            raise ValueError("locationの指定方法が間違っている")
        if self.is_leftward() and self.is_vertical():
            raise ValueError("locationの指定方法が間違っている")
        if self.location in {Location.RIGHTTOP, Location.LEFTTOP, Location.RIGHTBOTTOM, Location.LEFTBOTTOM}:
            raise ValueError("locationに手動で角を与える必要はない")

    def xs(self):
        return self.pt1[0], self.pt2[0]

    def ys(self):
        return self.pt1[1], self.pt2[1]

    def is_rightward(self):
        """
        ベクトルが右向きかどうか. 以後同様
        """
        start_x, end_x = self.xs()
        return start_x < end_x

    def is_leftward(self):
        start_x, end_x = self.xs()
        return start_x > end_x

    def is_upward(self):
        start_y, end_y = self.ys()
        return start_y > end_y

    def is_downward(self):
        start_y, end_y = self.ys()
        return start_y < end_y

    def is_vertical(self):
        return self.location == Location.RIGHT or self.location == Location.LEFT

    def is_horizontal(self):
        return self.location == Location.TOP or self.location == Location.BOTTOM

    def is_right(self):
        return self.location == Location.RIGHT

    def is_left(self):
        return self.location == Location.LEFT

    def is_top(self):
        return self.location == Location.TOP

    def is_bottom(self):
        return self.location == Location.BOTTOM
