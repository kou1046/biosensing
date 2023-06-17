from __future__ import annotations
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
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
class Obstacle:
    pt1: tuple[float, float]
    pt2: tuple[float, float]
    direction: Direction

    def xs(self):
        return self.pt1[0], self.pt2[0]

    def ys(self):
        return self.pt1[1], self.pt2[1]


@dataclass(frozen=True)
class Obstacles:
    value: list[Obstacle]

    def get_value(self):
        return self.value.copy()

    def subplots(self):
        fig, ax = plt.subplots()
        for obstacle in self.get_value():
            ax.plot(obstacle.xs(), obstacle.ys())
        return fig, ax


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

    @abstractmethod
    def input(self, x: float, y: float, t: float) -> float:
        ...


X = list[int]
Y = list[int]
XYIndices = list[X, Y]


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

        grid_row_last_index = self.calculate_grid_width() - 1
        grid_col_last_index = self.calculate_grid_height() - 1
        right_indices = [
            ([grid_row_last_index] * grid_col_last_index),
            list(range(0, grid_col_last_index)),
        ]
        left_indices = [
            ([0] * grid_col_last_index),
            list(range(0, grid_col_last_index)),
        ]
        top_indices = [
            list(range(0, grid_row_last_index)),
            ([0] * grid_row_last_index),
        ]
        bottom_indices = [
            list(range(0, grid_row_last_index)),
            ([grid_col_last_index] * grid_row_last_index),
        ]
        righttop_indices = [[grid_row_last_index], [0]]
        lefttop_indices = [[0], [0]]
        rightbottom_indices = [[grid_row_last_index], [grid_col_last_index]]
        leftbottom_indices = [[0], [grid_col_last_index]]

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
class WaveConditions:
    grid: Grid
    obstacles: Obstacles = None
    strain: Strain = None
    reflect_condition: ReflectCondition = ReflectCondition.NEUMANN

    def boundary_and_obstacle_indices(self) -> dict[Direction, XYIndices]:
        """
        障害物及び端付近のインデックスを取得する．
        """
        boundary_indices = self.grid.boundary_indices()
        if self.obstacles is None:
            return boundary_indices

        grid_row_last_index = self.grid.calculate_grid_width() - 1
        grid_col_last_index = self.grid.calculate_grid_height() - 1

        for obstacle in self.obstacles.get_value():
            index_xlim = (np.array(obstacle.xs()) / self.grid.h).astype(int)
            index_ylim = (np.array(obstacle.ys()) / self.grid.h).astype(int)


class Wave:
    def __init__(
        self,
        wave_conditions: WaveConditions,
        values: np.ndarray | None = None,
        pre_values: np.ndarray | None = None,
    ):
        if values is None:
            self.values = np.zeros(
                (
                    wave_conditions.grid.calculate_grid_width(),
                    wave_conditions.grid.calculate_grid_height(),
                )
            )
            self.pre_values = self.values.copy()
        else:
            self.values = values
            self.pre_values = pre_values
        self.conditions = wave_conditions

    def input_gauss(self, x0: float, y0: float, rad: float, A: float = 1.0):
        x = np.linspace(
            0,
            self.conditions.grid.width,
            int(self.conditions.grid.width // self.conditions.h),
        ).reshape(-1, 1)
        y = np.linspace(
            0,
            self.conditions.grid.height,
            int(self.conditions.grid.height // self.conditions.h),
        )
        input_values = A * np.exp(-((x - x0) ** 2) * rad**2) * np.exp(-((y - y0) ** 2) * rad**2)
        return Wave(self.conditions, self.values + input_values, self.pre_values + input_values)

    def update(self):
        uR = np.roll(self.values, -1, 1)
        uL = np.roll(self.values, 1, 1)
        uB = np.roll(self.values, -1, 0)
        uT = np.roll(self.values, 1, 0)

        # 一旦全ての点を拘束なしの条件でまとめて計算
        new_values = (
            2 * self.values - self.pre_values + self.conditions.grid.alpha * (uL + uR + uB + uT - 4 * self.values)
        )

        indices_items = self.conditions.boundary_and_obstacle_indices()

        X, Y = np.array(indices_items[Direction.RIGHT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X - 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 左端
        X, Y = np.array(indices_items[Direction.LEFT])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X + 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 上端
        X, Y = np.array(indices_items[Direction.TOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 下端
        X, Y = np.array(indices_items[Direction.BOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )

        # 左上端
        X, Y = np.array(indices_items[Direction.LEFTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右上
        X, Y = np.array(indices_items[Direction.RIGHTTOP])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
        )

        # 右下
        X, Y = np.array(indices_items[Direction.RIGHTBOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )

        # 左下
        X, Y = np.array(indices_items[Direction.LEFTBOTTOM])
        new_values[X, Y] = (
            2 * self.values[X, Y]
            - self.pre_values[X, Y]
            + self.conditions.grid.alpha
            * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
        )
        return Wave(self.conditions, new_values.copy(), self.values.copy())


grid = Grid(5, 3, 0.01, 0.005)

obstacle_list = [
    Obstacle((2, 1), (3, 1), Direction.BOTTOM),
    Obstacle((3, 1), (3, 2), Direction.RIGHT),
    Obstacle((3, 2), (2, 2), Direction.TOP),
    Obstacle((2, 2), (2, 1), Direction.LEFT),
]

obstacles = Obstacles(obstacle_list)

conditions = WaveConditions(grid, obstacles)
conditions.boundary_and_obstacle_indices()
wave = Wave(conditions)

fig, ax = plt.subplots()
wave = wave.input_gauss(0, 1.5, 3)
ims = []
for dt in np.arange(0, 3, 0.005):
    wave = wave.update()
    im = ax.imshow(wave.values.T)
    ims.append([im])
anim = ArtistAnimation(fig, ims, interval=30)
plt.show()
