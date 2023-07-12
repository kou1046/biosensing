from __future__ import annotations
from dataclasses import dataclass
from itertools import zip_longest

from ..types import Location, XYIndices
from ..obstacles import Obstacle
from ..walls import Wall
from ..strains import Strain


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
