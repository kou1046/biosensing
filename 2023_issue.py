from __future__ import annotations
import os
from matplotlib.animation import ArtistAnimation
import numpy as np
import matplotlib.pyplot as plt
import json

os.chdir(os.path.dirname(__file__))
from wavesimulator import Wave, Wall, Grid, Strain, Location, Obstacle, utils


class GaussStrain(Strain):
    def input(self, x: float, y: float, t: float) -> float:
        A = 1.0
        peak_time = 1.0
        sigma = 3.0
        return A * np.exp(-1 * (t - peak_time) ** 2 / 2 * sigma**2)


with open("rc_params.json", "r") as f:
    rc_params: dict = json.load(f)

for k, v in rc_params.items():
    plt.rc(k, **v)

L: float = 3.0
H: float = 0.01
DT: float = 0.005
OBSERVATION_TIME: float = 15.0

width = 2.0 * L
height = L
observation_point = (width, 3 * L / 4)

grid = Grid(width, height, H, DT)


pt1_1 = ((L / 2) - (L / 4), (L / 2) - (L / 4))
pt1_2 = ((L / 2) + (L / 4), (L / 2) - (L / 4))
pt1_3 = ((L / 2) + (L / 4), (L / 2) + (L / 4))
pt1_4 = ((L / 2) - (L / 4), (L / 2) + (L / 4))
walls = [
    Wall(pt1_1, pt1_2, Location.BOTTOM),
    Wall(pt1_2, pt1_3, Location.LEFT),
    Wall(pt1_3, pt1_4, Location.TOP),
    Wall(pt1_4, pt1_1, Location.RIGHT),
]
pt2_1 = (L, 0)
pt2_2 = (L, L / 2)
pt2_3 = (2.0 * L, L / 2)
walls_2 = [
    Wall(pt2_1, pt2_2, Location.RIGHT),
    Wall(pt2_2, pt2_3, Location.TOP),
]
obstacle_1 = Obstacle(walls)
obstacle_2 = Obstacle(walls_2)
obstacles = [obstacle_1, obstacle_2]

strain_1 = GaussStrain(Wall((0, 0), (0, height / 4), Location.LEFT))
strain_2 = GaussStrain(Wall((0, 0), (height / 4, 0), Location.TOP))
strains = [strain_1, strain_2]

""" 
wall座標図
                   2_1
 ________________________________
|                  |              |
|  1_1 ______ 1_2  |              |
|     |      |     |______________|
|     |      |    2_2             | 2_3
|     |______|                    |
|  1_4        1_3                 |
|                                 |
|_________________________________|

"""

wave = Wave(grid)

fig, axes = plt.subplots(2, 1)
grid_fig, grid_ax = utils.create_visualized_subplots(wave, obstacles, strains)
input_fig, input_ax = plt.subplots()

ims = []
times = np.arange(0, OBSERVATION_TIME + DT, DT)

observation_values: list[float] = []

for time in times:
    wave.update(obstacles, strains)
    observation_values.append(wave.get_value_by_cor(observation_point))
    im = axes[0].imshow(
        wave.get_values().T, "binary", vmin=-0.01, vmax=0.01, extent=[0, width, 0, height], origin="lower"
    )
    title = axes[0].text(
        0.5,
        1.01,
        f"Time = {round(time,2)}",
        ha="center",
        va="bottom",
        transform=axes[0].transAxes,
        fontsize="large",
    )
    im_2 = axes[1].plot(times[: len(observation_values)], observation_values, "k")
    ims.append([im, title] + im_2)

axes[0].invert_yaxis()
for obstacle in obstacles:
    axes[0].plot(obstacle.xs(), obstacle.ys(), color="k")
axes[0].scatter(*observation_point, color="r")
axes[0].annotate(
    "Observation point", xy=observation_point, xytext=(width + 1, height), xycoords="data", arrowprops=dict(color="k")
)
axes[1].set(title="Observation value", xlabel="Time")

grid_fig.legend(ncol=6, loc="upper center")
input_ax.plot(times, strain_1.input(0.0, 0.0, times), color="k")
input_ax.set(title="input", xlabel="Time")
anim = ArtistAnimation(fig, ims, interval=10)

plt.show()

# output_dir_name = "2023_result"
# os.makedirs(output_dir_name, exist_ok=True)
# anim.save(os.path.join(output_dir_name, "2023_issue.gif"))
# grid_fig.savefig(os.path.join(output_dir_name, "grid.png"))
# input_fig.savefig(os.path.join(output_dir_name, "input.png"))
