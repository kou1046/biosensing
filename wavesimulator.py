from __future__ import annotations
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from itertools import zip_longest
from typing import Callable
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


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


XIndecies = list[int]
YIndecies = list[int]
XYIndices = tuple[XIndecies, YIndecies]


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    h: float
    dt: float

    @property
    def alpha(self):
        return (self.dt / self.h) ** 2

    def __post_init__(self):
        if self.dt > self.h:
            raise ValueError("クーラン条件に基づき, 時間刻みdtは格子刻み幅hより小さくすること. ")

    def calculate_grid_row_num(self):
        return int(self.width // self.h)

    def calculate_grid_col_num(self):
        return int(self.height // self.h)

    def calculate_grid_index(self, cor: float):
        return int(cor // self.h)

    def boundary_indices(self) -> dict[Location, XYIndices]:
        """
        端付近のインデックス配列を取得する．
        """

        grid_row_num = self.calculate_grid_row_num()
        grid_col_num = self.calculate_grid_col_num()
        grid_row_last_index = grid_row_num - 1
        grid_col_last_index = grid_col_num - 1

        right_indices = (
            ([grid_row_last_index] * (grid_col_num - 2)),  # 角の処理のため，最初と最後は追加しない -> Xの要素数は grid_col_num - 2となる. 以降も同様．
            list(range(1, grid_col_num - 1)),
        )
        left_indices = (
            ([0] * (grid_col_num - 2)),
            list(range(1, grid_col_num - 1)),
        )
        top_indices = (
            list(range(1, grid_row_num - 1)),
            ([0] * (grid_row_num - 2)),
        )
        bottom_indices = (
            list(range(1, grid_row_num - 1)),
            ([grid_col_last_index] * (grid_row_num - 2)),
        )

        righttop_indices = ([grid_row_last_index], [0])
        lefttop_indices = ([0], [0])
        rightbottom_indices = ([grid_row_last_index], [grid_col_last_index])
        leftbottom_indices = ([0], [grid_col_last_index])

        return {
            location: indices
            for location, indices in zip(
                Location,
                [
                    right_indices,
                    left_indices,
                    top_indices,
                    bottom_indices,
                    righttop_indices,
                    lefttop_indices,
                    rightbottom_indices,
                    leftbottom_indices,
                ],
            )
        }


@dataclass(frozen=True)
class Wall:
    """_summary_
        障害物やひずみを表現するときに用いるクラス．
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


@dataclass(frozen=True)
class Obstacle:
    """_summary_
        障害物を表現するクラス. Wallクラスのファーストコレクション.
    Args:
        value: Wallの配列.
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

    def xlim(self):
        wall_xs = self.xs()
        return min(wall_xs), max(wall_xs)

    def ylim(self):
        wall_ys = np.array([wall.ys() for wall in self.walls]).ravel()
        return min(wall_ys), max(wall_ys)

    def locations(self):
        return [wall.location for wall in self.walls]

    def grid_indices(self, grid: Grid) -> dict[Location, XYIndices]:
        """
        障害物の格子インデックスを取得する．
        """

        wall_xmin, wall_xmax = self.xlim()
        if wall_xmin < 0 or wall_xmax > grid.width:
            raise ValueError("格子の幅が足りない")
        wall_ymin, wall_ymax = self.ylim()
        if wall_ymin < 0 or wall_ymax > grid.height:
            raise ValueError("格子の高さが足りない")

        grid_row_last_index = grid.calculate_grid_row_num() - 1
        grid_col_last_index = grid.calculate_grid_col_num() - 1

        wall_indecies: dict[Location, XYIndices] = {location: ([], []) for location in Location}

        for wall, next_wall in zip_longest(self.walls, self.walls[1:]):
            if next_wall is None:
                next_wall = self.walls[0]

            if wall.is_horizontal():
                ys = wall.ys()
                assert ys[0] == ys[1], "障害物が並行でない"
                obstalce_xlim = sorted(wall.xs())
                wall_index_xlim = [grid.calculate_grid_index(x) for x in obstalce_xlim]
                xmin, xmax = wall_index_xlim
                wall_indicies_x = list(range(xmin, xmax + 1))
                wall_indicies_y = [grid.calculate_grid_index(ys[0])] * len(wall_indicies_x)

            if wall.is_vertical():
                xs = wall.xs()
                assert xs[0] == xs[1], "障害物が垂直でない"
                obstalce_ylim = sorted(wall.ys())
                wall_index_ylim = [grid.calculate_grid_index(y) for y in obstalce_ylim]
                ymin, ymax = wall_index_ylim
                wall_indicies_y = list(range(ymin, ymax + 1))
                wall_indicies_x = [grid.calculate_grid_index(xs[0])] * len(wall_indicies_y)

            if wall.is_right():
                X, Y = wall_indecies[Location.RIGHT]
            if wall.is_left():
                X, Y = wall_indecies[Location.LEFT]
            if wall.is_top():
                X, Y = wall_indecies[Location.TOP]
            if wall.is_bottom():
                X, Y = wall_indecies[Location.BOTTOM]

            X.extend(wall_indicies_x)
            Y.extend(wall_indicies_y)

            if wall.is_right() and next_wall.is_top() and next_wall.is_leftward():
                """
                ___
                   o <- この角
                   |
                   |
                """
                X, Y = wall_indecies[Location.RIGHTTOP]
                X.append(wall_indicies_x[0])
                Y.append(wall_indicies_y[0])

            if wall.is_left() and next_wall.is_rightward() and next_wall.is_bottom():
                """
                |
                |
                o____
                ↑ この角
                """
                X, Y = wall_indecies[Location.LEFTBOTTOM]
                X.append(wall_indicies_x[0])
                Y.append(wall_indicies_y[-1])

            if wall.is_top() and next_wall.is_downward() and next_wall.is_left():
                """
                ___
                o  <- この角
                |
                |
                """

                X, Y = wall_indecies[Location.LEFTTOP]
                X.append(wall_indicies_x[0])
                Y.append(wall_indicies_y[0])
            if wall.is_bottom() and next_wall.is_upward() and next_wall.is_right():
                """
                   |
                   |
                ___o <- この核
                """
                X, Y = wall_indecies[Location.RIGHTBOTTOM]
                X.append(wall_indicies_x[-1])
                Y.append(wall_indicies_y[0])

        return wall_indecies


class Strain(metaclass=ABCMeta):
    def __init__(
        self,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        location: Location,
    ):
        self.pt1 = pt1
        self.pt2 = pt2
        self.location = location

    def grid_indices(self, grid: Grid):
        """
        ひずみの格子インデックスを取得する．
        """
        pass

    @abstractmethod
    def input(self, x: float, y: float, t: float) -> float:
        ...


class Wave:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.values: np.ndarray = np.zeros(
            (
                grid.calculate_grid_row_num(),
                grid.calculate_grid_col_num(),
            )
        )
        self.pre_values = self.values.copy()
        self.time = 0.0

    def get_values(self):
        return self.values.copy()

    def input_gauss(self, x0: float, y0: float, rad: float, A: float = 1.0):
        x = np.linspace(
            0,
            self.grid.width,
            self.grid.calculate_grid_row_num(),
        ).reshape(-1, 1)
        y = np.linspace(
            0,
            self.grid.height,
            self.grid.calculate_grid_col_num(),
        )
        input_values = A * np.exp(-((x - x0) ** 2) * rad**2) * np.exp(-((y - y0) ** 2) * rad**2)
        self.values = self.values + input_values
        self.pre_values = self.pre_values + input_values

    def update(self, obstacle: Obstacle | None = None, strain: Strain | None = None):
        uR = np.roll(self.values, -1, 1)
        uL = np.roll(self.values, 1, 1)
        uB = np.roll(self.values, -1, 0)
        uT = np.roll(self.values, 1, 0)

        # 一旦全ての点を拘束なしの条件でまとめて計算
        new_values = 2 * self.values - self.pre_values + self.grid.alpha * (uL + uR + uB + uT - 4 * self.values)

        # 端のインデックス群を取得
        indices_items = self.grid.boundary_indices()

        # 障害物が与えられれば障害物のインデックス群を取得して結合
        if obstacle is not None:
            obstacle_grid_indices = obstacle.grid_indices(self.grid)
            for location in Location:
                X, Y = indices_items[location]
                wall_X, wall_Y = obstacle_grid_indices[location]
                X.extend(wall_X)
                Y.extend(wall_Y)

        # ひずみが与えられればひずみのインデックス群を取得して結合
        if strain is not None:
            strain_grid_indices = strain.grid_indices(self.grid)
            for location in Location:
                X, Y = indices_items[location]
                strain_X, strain_Y = strain_grid_indices[location]
                X.extend(strain_X)
                Y.extend(strain_Y)

        X, Y = np.array(indices_items[Location.RIGHT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (2 * self.values[X - 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = X + 1 < self.values.shape[0]
        new_values[X[o_idxes] + 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処

        # 左端
        X, Y = np.array(indices_items[Location.LEFT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (2 * self.values[X + 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = X > 0
        new_values[X[o_idxes] - 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処理

        # 上端
        X, Y = np.array(indices_items[Location.TOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = Y > 0
        new_values[X[o_idxes], Y[o_idxes] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 下端
        X, Y = np.array(indices_items[Location.BOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )
        o_idxes = Y + 1 < self.values.shape[1]
        new_values[X[o_idxes], Y[o_idxes] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 左上端
        X, Y = np.array(indices_items[Location.LEFTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右上
        X, Y = np.array(indices_items[Location.RIGHTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右下
        X, Y = np.array(indices_items[Location.RIGHTBOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )

        # 左下
        X, Y = np.array(indices_items[Location.LEFTBOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )

        self.pre_values = self.values
        self.values = new_values
        self.time += self.grid.dt


width = 5
height = 5
h = 0.01
dt = 0.005
grid = Grid(width, height, h, dt)

wall_xs = [1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1]
wall_ys = [2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 2]
wall_list = [
    Obstacle((x0, y0), (x1, y1), location)
    for x0, y0, x1, y1, location in (
        zip(
            wall_xs,
            wall_ys,
            wall_xs[1:],
            wall_ys[1:],
            [
                Location.BOTTOM,
                Location.RIGHT,
                Location.BOTTOM,
                Location.LEFT,
                Location.BOTTOM,
                Location.LEFT,
                Location.TOP,
                Location.LEFT,
                Location.TOP,
                Location.RIGHT,
                Location.TOP,
                Location.RIGHT,
            ],
        )
    )
]
obstacle = Obstacle(wall_list)

wave = Wave(grid)
wave.input_gauss(0, 0.5, 3)
ims = []
fig, ax = plt.subplots()
for time in np.arange(0, 10 + dt, dt):
    wave.update(obstacle)
    ims.append([ax.imshow(wave.values.T, "binary")])
anim = ArtistAnimation(fig, ims, interval=20)
plt.show()
