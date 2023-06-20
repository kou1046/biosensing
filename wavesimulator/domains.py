from __future__ import annotations
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from itertools import zip_longest
from enum import Enum
import numpy as np


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


class Strain(metaclass=ABCMeta):
    def __init__(self, wall: Wall):
        self.wall = wall

    @abstractmethod
    def input(self, x: float, y: float, t: float) -> float:
        ...


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
        return int(self.width / self.h)

    def calculate_grid_col_num(self):
        return int(self.height / self.h)

    def calculate_grid_num(self, cor: float):
        return int(cor / self.h)

    def calculate_boundary_indices(self) -> dict[Location, XYIndices]:
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

    def calculate_wall_indices(self, wall: Wall) -> XYIndices:
        """
        引数として与えられた壁(wall)の格子インデックスを取得する．
        """
        wall_xmin, wall_xmax = sorted(wall.xs())
        if wall_xmin < 0 or wall_xmax > self.width:
            raise ValueError("格子の幅が足りない")
        wall_ymin, wall_ymax = sorted(wall.ys())
        if wall_ymin < 0 or wall_ymax > self.height:
            raise ValueError("格子の高さが足りない")

        if wall.is_horizontal():
            ys = wall.ys()
            assert ys[0] == ys[1], "障害物が並行でない"
            obstalce_xlim = sorted(wall.xs())
            wall_index_xlim = [self.calculate_grid_num(x) for x in obstalce_xlim]
            xmin, xmax = wall_index_xlim
            wall_indices_x = list(range(xmin, xmax))
            wall_indices_y = [self.calculate_grid_num(ys[0])] * len(wall_indices_x)

        if wall.is_vertical():
            xs = wall.xs()
            assert xs[0] == xs[1], "障害物が垂直でない"
            obstalce_ylim = sorted(wall.ys())
            wall_index_ylim = [self.calculate_grid_num(y) for y in obstalce_ylim]
            ymin, ymax = wall_index_ylim
            wall_indices_y = list(range(ymin, ymax))
            wall_indices_x = [self.calculate_grid_num(xs[0])] * len(wall_indices_y)

        return (wall_indices_x, wall_indices_y)

    def calculate_obstacle_indices(self, obstacle: Obstacle) -> dict[Location, XYIndices]:
        """
        引数として与えられた障害物(Obstacle)の格子インデックスを取得する．
        """

        grid_row_last_index = self.calculate_grid_row_num() - 1
        grid_col_last_index = self.calculate_grid_col_num() - 1

        wall_indices: dict[Location, XYIndices] = {location: ([], []) for location in Location}

        for wall, next_wall in zip_longest(obstacle.walls, obstacle.walls[1:]):
            if next_wall is None:
                next_wall = obstacle.walls[0]

            wall_indices_x, wall_indices_y = self.calculate_wall_indices(wall)

            if wall_indices_x[-1] == grid_row_last_index and wall.is_top():  # 障害物が壁に隣接した場合，端の2つを消去しないとエラーが出る
                """
                __o ← この時, oとその左のインデックスを消去
                  |
                  |
                """
                X, Y = wall_indices[Location.RIGHTTOP]
                X.append(wall_indices_x[-1])
                Y.append(wall_indices_y[0])
                del wall_indices_x[-1]
                del wall_indices_y[-1]

            if wall.is_right():
                X, Y = wall_indices[Location.RIGHT]
            if wall.is_left():
                X, Y = wall_indices[Location.LEFT]
            if wall.is_top():
                X, Y = wall_indices[Location.TOP]
            if wall.is_bottom():
                X, Y = wall_indices[Location.BOTTOM]

            X.extend(wall_indices_x)
            Y.extend(wall_indices_y)

            if (wall.is_right() and next_wall.is_top() and next_wall.is_leftward()) or (
                wall_indices_y[0] == 0 and wall.is_right()
            ):
                """
                ___
                   o <- この角
                   |
                   |
                """
                X, Y = wall_indices[Location.RIGHTTOP]
                X.append(wall_indices_x[0])
                Y.append(wall_indices_y[0])

            if wall.is_left() and next_wall.is_rightward() and next_wall.is_bottom():
                """
                |
                |
                o____
                ↑ この角
                """
                X, Y = wall_indices[Location.LEFTBOTTOM]
                X.append(wall_indices_x[0])
                Y.append(wall_indices_y[-1])

            if wall.is_top() and next_wall.is_downward() and next_wall.is_left():
                """
                ___
                o  <- この角
                |
                |
                """

                X, Y = wall_indices[Location.LEFTTOP]
                X.append(wall_indices_x[0])
                Y.append(wall_indices_y[0])
            if wall.is_bottom() and next_wall.is_upward() and next_wall.is_right():
                """
                   |
                   |
                ___o <- この核
                """
                X, Y = wall_indices[Location.RIGHTBOTTOM]
                X.append(wall_indices_x[-1])
                Y.append(wall_indices_y[0])

        return wall_indices

    def calculate_strain_indices(self, strain: Strain) -> dict[Location, XYIndices]:
        """
        引数として与えられたひずみ(Strain)の格子インデックスを取得する.
        """
        strain_indices: dict[Location, XYIndices] = {location: ([], []) for location in Location}

        x_indices, y_indices = self.calculate_wall_indices(strain.wall)
        if strain.wall.is_right():
            X, Y = strain_indices[Location.RIGHT]
        if strain.wall.is_left():
            X, Y = strain_indices[Location.LEFT]
        if strain.wall.is_top():
            X, Y = strain_indices[Location.TOP]
        if strain.wall.is_bottom():
            X, Y = strain_indices[Location.BOTTOM]
        X.extend(x_indices)
        Y.extend(y_indices)

        return strain_indices


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

    def get_value_by_cor(self, point: tuple[float, float]):
        x_index, y_index = [self.grid.calculate_grid_num(cor) - 1 for cor in point]
        return self.values[x_index, y_index]

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

    def update(self, obstacles: list[Obstacle] | None = None, strains: list[Strain] | None = None):
        uR = np.roll(self.values, -1, 1)
        uL = np.roll(self.values, 1, 1)
        uB = np.roll(self.values, -1, 0)
        uT = np.roll(self.values, 1, 0)

        # 一旦全ての点を拘束なしの条件でまとめて計算
        new_values = 2 * self.values - self.pre_values + self.grid.alpha * (uL + uR + uB + uT - 4 * self.values)

        # 端のインデックス群を取得
        indices_items = self.grid.calculate_boundary_indices()

        # 障害物が与えられれば障害物のインデックス群を取得して結合
        if obstacles is not None:
            for obstacle in obstacles:
                obstacle_grid_indices = self.grid.calculate_obstacle_indices(obstacle)
                for location in Location:
                    X, Y = indices_items[location]
                    wall_X, wall_Y = obstacle_grid_indices[location]
                    X.extend(wall_X)
                    Y.extend(wall_Y)

        # 右端
        X, Y = np.array(indices_items[Location.RIGHT])
        if len(X):
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
        if len(X):
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
        if len(X):
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
        if len(X):
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
        if len(X):
            new_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X > 0, Y > 0
            new_values[X[o_idxes_1] - 1, Y[o_idxes_1]] = 0
            new_values[X[o_idxes_2], Y[o_idxes_2] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 右上
        X, Y = np.array(indices_items[Location.RIGHTTOP])
        if len(X):
            new_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X + 1 < self.values.shape[0], Y > 0
            new_values[X[o_idxes_1] + 1, Y[o_idxes_1]] = 0
            new_values[X[o_idxes_2], Y[o_idxes_2] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 右下
        X, Y = np.array(indices_items[Location.RIGHTBOTTOM])
        if len(X):
            new_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X + 1 < self.values.shape[0], Y + 1 < self.values.shape[1]
            new_values[X[o_idxes_1] + 1, Y[o_idxes_1]] = 0
            new_values[X[o_idxes_2], Y[o_idxes_2] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # 左下
        X, Y = np.array(indices_items[Location.LEFTBOTTOM])
        if len(X):
            new_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X > 0, Y + 1 < self.values.shape[1]
            new_values[X[o_idxes_1] - 1, Y[o_idxes_1]] = 0
            new_values[X[o_idxes_2], Y[o_idxes_2] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

        # ひずみが与えられればひずみ拘束条件の計算を加える (現時点で左と上のひずみのみ)
        if strains is not None:
            strain_indices_items = {location: ([], []) for location in Location}
            for strain in strains:
                strain_indices_item = self.grid.calculate_strain_indices(strain)
                for location in Location:
                    X, Y = strain_indices_items[location]
                    strain_X, strain_Y = strain_indices_item[location]
                    X.extend(strain_X)
                    Y.extend(strain_Y)

            X, Y = np.array(strain_indices_items[Location.LEFT])
            if len(X):
                new_values[X, Y] = (
                    2 * self.values[X, Y]
                    - self.pre_values[X, Y]
                    + self.grid.alpha
                    * (
                        self.values[X + 1, Y]
                        + self.values[X, Y + 1]
                        + self.values[X, Y - 1]
                        - 4 * self.values[X, Y]
                        - 2 * self.grid.h * strain.input(X, Y, self.time)
                    )
                )

            X, Y = np.array(strain_indices_items[Location.TOP])
            if len(X):
                new_values[X, Y] = (
                    2 * self.values[X, Y]
                    - self.pre_values[X, Y]
                    + self.grid.alpha
                    * (
                        self.values[X + 1, Y]
                        + self.values[X - 1, Y]
                        + self.values[X, Y + 1]
                        - 4 * self.values[X, Y]
                        - 2 * self.grid.h * strain.input(X, Y, self.time)
                    )
                )

        self.pre_values = self.values
        self.values = new_values
        self.time += self.grid.dt
