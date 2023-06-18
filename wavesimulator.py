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


class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3
    RIGHTTOP = 4
    LEFTTOP = 5
    RIGHTBOTTOM = 6
    LEFTBOTTOM = 7


@dataclass(frozen=True)
class PositiveFloatXY:
    x: float
    y: float

    def __post_init__(self):
        if self.x < 0 or self.y < 0:
            raise ValueError("正の値しか認めない")
        if not isinstance(float, self.x) or not isinstance(float, self.y):
            raise ValueError("floatに変換する必要がある")


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

    def calculate_grid_width(self):
        return int(self.width // self.h)

    def calculate_grid_height(self):
        return int(self.height // self.h)

    def boundary_indices(self) -> dict[Direction, XYIndices]:
        """
        端付近のインデックス配列を取得する．
        """

        grid_width = self.calculate_grid_width()
        grid_height = self.calculate_grid_height()
        grid_row_last_index = grid_width - 1
        grid_col_last_index = grid_height - 1

        right_indices = (
            ([grid_row_last_index] * (grid_height - 2)),  # 角の処理のため，最初と最後は追加しない -> Xの要素数は grid_height - 2となる. 以降も同様．
            list(range(1, grid_height - 1)),
        )
        left_indices = (
            ([0] * (grid_height - 2)),
            list(range(1, grid_height - 1)),
        )
        top_indices = (
            list(range(1, grid_width - 1)),
            ([0] * (grid_width - 2)),
        )
        bottom_indices = (
            list(range(1, grid_width - 1)),
            ([grid_col_last_index] * (grid_width - 2)),
        )

        righttop_indices = ([grid_row_last_index], [0])
        lefttop_indices = ([0], [0])
        rightbottom_indices = ([grid_row_last_index], [grid_col_last_index])
        leftbottom_indices = ([0], [grid_col_last_index])

        return {
            direction: indices
            for direction, indices in zip(
                Direction,
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
class Obstacle:
    pt1: tuple[float, float]
    pt2: tuple[float, float]
    direction: Direction

    def __post_init__(self):
        if self.is_upward() and self.is_horizontal():
            raise ValueError("directionの指定方法が間違っている")
        if self.is_downward() and self.is_horizontal():
            raise ValueError("directionの指定方法が間違っている")
        if self.is_rightward() and self.is_vertical():
            raise ValueError("directionの指定方法が間違っている")
        if self.is_leftward() and self.is_vertical():
            raise ValueError("directionの指定方法が間違っている")
        if self.direction in {Direction.RIGHTTOP, Direction.LEFTTOP, Direction.RIGHTBOTTOM, Direction.LEFTBOTTOM}:
            raise ValueError("directionに手動で角を与える必要はない")

    def xs(self):
        return self.pt1[0], self.pt2[0]

    def ys(self):
        return self.pt1[1], self.pt2[1]

    def is_rightward(self):
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
        return self.direction == Direction.RIGHT or self.direction == Direction.LEFT

    def is_horizontal(self):
        return self.direction == Direction.TOP or self.direction == Direction.BOTTOM

    def is_right(self):
        return self.direction == Direction.RIGHT

    def is_left(self):
        return self.direction == Direction.LEFT

    def is_top(self):
        return self.direction == Direction.TOP

    def is_bottom(self):
        return self.direction == Direction.BOTTOM


@dataclass(frozen=True)
class Obstacles:
    value: list[Obstacle]

    def get_value(self):
        return self.value.copy()

    def xs(self):
        return np.array([obstacle.xs() for obstacle in self.value]).ravel().tolist()

    def ys(self):
        return np.array([obstacle.ys() for obstacle in self.value]).ravel().tolist()

    def xlim(self):
        obstacle_xs = self.xs()
        return min(obstacle_xs), max(obstacle_xs)

    def ylim(self):
        obstacle_ys = np.array([obstacle.ys() for obstacle in self.value]).ravel()
        return min(obstacle_ys), max(obstacle_ys)

    def directions(self):
        return [obstacle.direction for obstacle in self.value]

    def create_fig_ax(self):
        fig, ax = plt.subplots()
        for obstacle in self.get_value():
            ax.plot(obstacle.xs(), obstacle.ys())
        return fig, ax

    def grid_indices(self, grid: Grid) -> dict[Direction, XYIndices]:
        """
        障害物の格子インデックスを取得する．
        """

        obstacle_xmin, obstacle_xmax = self.xlim()
        if obstacle_xmin < 0 or obstacle_xmax > grid.width:
            raise ValueError("格子の幅が足りない")
        obstacle_ymin, obstacle_ymax = self.ylim()
        if obstacle_ymin < 0 or obstacle_ymax > grid.height:
            raise ValueError("格子の高さが足りない")

        grid_row_last_index = grid.calculate_grid_width() - 1
        grid_col_last_index = grid.calculate_grid_height() - 1

        obstacle_indecies: dict[Direction, XYIndices] = {direction: ([], []) for direction in Direction}

        for obstacle, next_obstacle in zip_longest(self.value, self.value[1:]):
            if next_obstacle is None:
                next_obstacle = self.value[0]

            if obstacle.is_horizontal():
                ys = obstacle.ys()
                assert ys[0] == ys[1], "障害物が並行でない"
                obstacle_index_xlim = (np.array(sorted(obstacle.xs())) // grid.h).astype(int)
                xmin, xmax = obstacle_index_xlim
                obstacle_indicies_x = list(range(xmin, xmax + 1))
                obstacle_indicies_y = [int(ys[0] // grid.h)] * len(obstacle_indicies_x)

            if obstacle.is_vertical():
                xs = obstacle.xs()
                assert xs[0] == xs[1], "障害物が垂直でない"
                obstacle_index_ylim = (np.array(sorted(obstacle.ys())) // grid.h).astype(int)
                ymin, ymax = obstacle_index_ylim
                obstacle_indicies_y = list(range(ymin, ymax + 1))
                obstacle_indicies_x = [int(xs[0] // grid.h)] * len(obstacle_indicies_y)

            if obstacle.is_right():
                X, Y = obstacle_indecies[Direction.RIGHT]
            if obstacle.is_left():
                X, Y = obstacle_indecies[Direction.LEFT]
            if obstacle.is_top():
                X, Y = obstacle_indecies[Direction.TOP]
            if obstacle.is_bottom():
                X, Y = obstacle_indecies[Direction.BOTTOM]

            X.extend(obstacle_indicies_x)
            Y.extend(obstacle_indicies_y)

            if obstacle.is_right() and next_obstacle.is_top() and next_obstacle.is_leftward():
                """
                ___
                   o <- この角
                   |
                   |
                """
                X, Y = obstacle_indecies[Direction.RIGHTTOP]
                X.append(obstacle_indicies_x[0])
                Y.append(obstacle_indicies_y[0])

            if obstacle.is_left() and next_obstacle.is_rightward() and next_obstacle.is_bottom():
                """
                |
                |
                o____
                ↑ この角
                """
                X, Y = obstacle_indecies[Direction.LEFTBOTTOM]
                X.append(obstacle_indicies_x[0])
                Y.append(obstacle_indicies_y[-1])

            if obstacle.is_top() and next_obstacle.is_downward() and next_obstacle.is_left():
                """
                ___
                o  <- この角
                |
                |
                """

                X, Y = obstacle_indecies[Direction.LEFTTOP]
                X.append(obstacle_indicies_x[0])
                Y.append(obstacle_indicies_y[0])
            if obstacle.is_bottom() and next_obstacle.is_upward() and next_obstacle.is_right():
                """
                   |
                   |
                ___o <- この核
                """
                X, Y = obstacle_indecies[Direction.RIGHTBOTTOM]
                X.append(obstacle_indicies_x[-1])
                Y.append(obstacle_indicies_y[0])

        return obstacle_indecies


class Strain(metaclass=ABCMeta):
    def __init__(
        self,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        direction: Direction,
    ):
        self.pt1 = pt1
        self.pt2 = pt2
        self.direction = direction

    def grid_indices(self):
        pass

    @abstractmethod
    def input(self, x: float, y: float, t: float) -> float:
        ...


class Wave:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.values: np.ndarray = np.zeros(
            (
                grid.calculate_grid_width(),
                grid.calculate_grid_height(),
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
            int(self.grid.width // self.grid.h),
        ).reshape(-1, 1)
        y = np.linspace(
            0,
            self.grid.height,
            int(self.grid.height // self.grid.h),
        )
        input_values = A * np.exp(-((x - x0) ** 2) * rad**2) * np.exp(-((y - y0) ** 2) * rad**2)
        self.values = self.values + input_values
        self.pre_values = self.pre_values + input_values

    def update(self, obstacles: Obstacles | None = None, strain: Strain | None = None):
        uR = np.roll(self.values, -1, 1)
        uL = np.roll(self.values, 1, 1)
        uB = np.roll(self.values, -1, 0)
        uT = np.roll(self.values, 1, 0)

        # 一旦全ての点を拘束なしの条件でまとめて計算
        new_values = 2 * self.values - self.pre_values + self.grid.alpha * (uL + uR + uB + uT - 4 * self.values)

        # 端のインデックス群を取得
        indices_items = self.grid.boundary_indices()

        # 障害物が与えられれば障害物のインデックス群を取得して結合
        if obstacles is not None:
            obstacle_grid_indices = obstacles.grid_indices(self.grid)
            for direction in Direction:
                X, Y = indices_items[direction]
                obstacle_X, obstacle_Y = obstacle_grid_indices[direction]
                X.extend(obstacle_X)
                Y.extend(obstacle_Y)

        # ひずみが与えられればひずみのインデックス群を取得して結合
        if strain is not None:
            strain_grid_indices = strain.grid_indices(self.grid)
            for direction in Direction:
                X, Y = indices_items[direction]
                strain_X, strain_Y = strain_grid_indices[direction]
                X.extend(strain_X)
                Y.extend(strain_Y)

        X, Y = np.array(indices_items[Direction.RIGHT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (2 * self.values[X - 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = X + 1 < self.values.shape[0]
        new_values[X[o_idxes] + 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処

        # 左端
        X, Y = np.array(indices_items[Direction.LEFT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (2 * self.values[X + 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = X > 0
        new_values[X[o_idxes] - 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処理

        # 上端
        X, Y = np.array(indices_items[Direction.TOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )
        o_idxes = Y > 0
        new_values[X[o_idxes], Y[o_idxes] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 下端
        X, Y = np.array(indices_items[Direction.BOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )
        o_idxes = Y + 1 < self.values.shape[1]
        new_values[X[o_idxes], Y[o_idxes] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 左上端
        X, Y = np.array(indices_items[Direction.LEFTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右上
        X, Y = np.array(indices_items[Direction.RIGHTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右下
        X, Y = np.array(indices_items[Direction.RIGHTBOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )

        # 左下
        X, Y = np.array(indices_items[Direction.LEFTBOTTOM])
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

obstacle_xs = [1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1]
obstacle_ys = [2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 2]
obstacle_list = [
    Obstacle((x0, y0), (x1, y1), direction)
    for x0, y0, x1, y1, direction in (
        zip(
            obstacle_xs,
            obstacle_ys,
            obstacle_xs[1:],
            obstacle_ys[1:],
            [
                Direction.BOTTOM,
                Direction.RIGHT,
                Direction.BOTTOM,
                Direction.LEFT,
                Direction.BOTTOM,
                Direction.LEFT,
                Direction.TOP,
                Direction.LEFT,
                Direction.TOP,
                Direction.RIGHT,
                Direction.TOP,
                Direction.RIGHT,
            ],
        )
    )
]
obstacles = Obstacles(obstacle_list)

wave = Wave(grid)
wave.input_gauss(0, 0.5, 3)
ims = []
fig, ax = plt.subplots()
for time in np.arange(0, 10 + dt, dt):
    wave.update(obstacles)
    ims.append([ax.imshow(wave.values.T, "binary")])
anim = ArtistAnimation(fig, ims, interval=20)
plt.show()
