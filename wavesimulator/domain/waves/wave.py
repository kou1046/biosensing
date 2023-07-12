from __future__ import annotations
import numpy as np

from ..grids import Grid
from ..strains import Strain
from ..obstacles import Obstacle
from ..types import Location, XYIndices


class Wave:
    def __init__(self, grid: Grid):
        """
        波．以下に列挙するメソッドは自身の状態を変更する副作用を持ち，返り値はない
            update
            input_gauss
        """
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
        """
        返り値なし．このメソッドは自身の状態を変更する．
        """

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
        self._reflect_from_right(new_values, indices_items[Location.RIGHT])

        # 左端
        self._reflect_from_left(new_values, indices_items[Location.LEFT])

        # 上端
        self._reflect_from_top(new_values, indices_items[Location.TOP])

        # 下端
        self._reflect_from_bottom(new_values, indices_items[Location.BOTTOM])

        # 左上端
        self._reflect_from_lefttop(new_values, indices_items[Location.LEFTTOP])

        # 右上
        self._reflect_from_righttop(new_values, indices_items[Location.RIGHTTOP])

        # 右下
        self._reflect_from_rightbottom(new_values, indices_items[Location.RIGHTBOTTOM])

        # 左下
        self._reflect_from_leftbottom(new_values, indices_items[Location.LEFTBOTTOM])

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

            # 左のひずみ
            self._reflect_from_leftstrain(new_values, strain_indices_items[Location.LEFT], strain)
            # 上のひずみ
            self._reflect_from_topstrain(new_values, strain_indices_items[Location.TOP], strain)

        self.pre_values = self.values
        self.values = new_values
        self.time += self.grid.dt

    def _reflect_from_right(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        """
        このメソッドを含む reflect ~ のメソッドは与えられた引数 (inplaced_values) を変更する.
        """
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha
                * (2 * self.values[X - 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes = X + 1 < self.values.shape[0]
            inplaced_values[X[o_idxes] + 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_left(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha
                * (2 * self.values[X + 1, Y] + self.values[X, Y - 1] + self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes = X > 0
            inplaced_values[X[o_idxes] - 1, Y[o_idxes]] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_top(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha
                * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes = Y > 0
            inplaced_values[X[o_idxes], Y[o_idxes] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_bottom(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha
                * (self.values[X - 1, Y] + self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
            )
            o_idxes = Y + 1 < self.values.shape[1]
            inplaced_values[X[o_idxes], Y[o_idxes] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_righttop(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            X, Y = np.array(xyindices)
            o_idxes_1, o_idxes_2 = X + 1 < self.values.shape[0], Y > 0
            X, Y = np.array(xyindices)
            inplaced_values[X[o_idxes_1] + 1, Y[o_idxes_1]] = 0
            inplaced_values[X[o_idxes_2], Y[o_idxes_2] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_lefttop(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y + 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X > 0, Y > 0
            inplaced_values[X[o_idxes_1] - 1, Y[o_idxes_1]] = 0
            inplaced_values[X[o_idxes_2], Y[o_idxes_2] - 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_rightbottom(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X - 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X + 1 < self.values.shape[0], Y + 1 < self.values.shape[1]
            inplaced_values[X[o_idxes_1] + 1, Y[o_idxes_1]] = 0
            inplaced_values[X[o_idxes_2], Y[o_idxes_2] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_leftbottom(self, inplaced_values: np.ndarray, xyindices: XYIndices):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
                2 * self.values[X, Y]
                - self.pre_values[X, Y]
                + self.grid.alpha * (2 * self.values[X + 1, Y] + 2 * self.values[X, Y - 1] - 4 * self.values[X, Y])
            )
            o_idxes_1, o_idxes_2 = X > 0, Y + 1 < self.values.shape[1]
            inplaced_values[X[o_idxes_1] - 1, Y[o_idxes_1]] = 0
            inplaced_values[X[o_idxes_2], Y[o_idxes_2] + 1] = 0  # 障害物内部に波が侵入しないようにする処理

    def _reflect_from_leftstrain(self, inplaced_values: np.ndarray, xyindices: XYIndices, strain: Strain):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
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

    def _reflect_from_topstrain(self, inplaced_values: np.ndarray, xyindices: XYIndices, strain: Strain):
        X, Y = np.array(xyindices)
        if len(X):
            inplaced_values[X, Y] = (
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
