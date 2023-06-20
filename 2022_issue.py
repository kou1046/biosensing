from __future__ import annotations
import os
from matplotlib.animation import ArtistAnimation
import numpy as np
import matplotlib.pyplot as plt
import json

os.chdir(os.path.dirname(__file__))
from wavesimulator import Wave, Wall, Grid, Strain, Location, Obstacle, utils


class CosineStrain(Strain):
    def __init__(self, wall: Wall):
        super().__init__(wall)

    def input(self, x: float, y: float, t: float) -> float:
        f = 3
        return np.cos(2 * np.pi * f * t) if t <= 3 * (1 / f) else 0


with open("rc_params.json", "r") as f:
    rc_params: dict = json.load(f)

for k, v in rc_params.items():
    plt.rc(k, **v)

WIDTH = 5
HEIGHT = 1
H: float = 0.01
DT: float = 0.005
OBSERVATION_TIME: float = 15.0

grid = Grid(WIDTH, HEIGHT, H, DT)

obstacle_height_1 = 0.4
obstacle_height_2 = 0.6
obstacle_width = 0.8
pt1 = (WIDTH / 2 - obstacle_width / 2, HEIGHT / 2 - obstacle_height_1 / 2)
pt2 = (WIDTH / 2, HEIGHT / 2 - obstacle_height_1 / 2)
pt3 = (WIDTH / 2, HEIGHT / 2 - obstacle_height_2 / 2)
pt4 = (WIDTH / 2 + obstacle_width / 2, HEIGHT / 2 - obstacle_height_2 / 2)
pt5 = (WIDTH / 2 + obstacle_width / 2, HEIGHT / 2 + obstacle_height_2 / 2)
pt6 = (WIDTH / 2, HEIGHT / 2 + obstacle_height_2 / 2)
pt7 = (WIDTH / 2, HEIGHT / 2 + obstacle_height_1 / 2)
pt8 = (WIDTH / 2 - obstacle_width / 2, HEIGHT / 2 + obstacle_height_1 / 2)
pt9 = (WIDTH / 2 - obstacle_width / 2, HEIGHT / 2 - obstacle_height_1 / 2)
walls = [
    Wall(pt1, pt2, Location.BOTTOM),
    Wall(pt2, pt3, Location.RIGHT),
    Wall(pt3, pt4, Location.BOTTOM),
    Wall(pt4, pt5, Location.LEFT),
    Wall(pt5, pt6, Location.TOP),
    Wall(pt6, pt7, Location.RIGHT),
    Wall(pt7, pt8, Location.TOP),
    Wall(pt8, pt9, Location.RIGHT),
]

obstacle = Obstacle(walls)
obstacles = [obstacle]

strain = CosineStrain(Wall((0, HEIGHT / 2 + 0.2), (0, HEIGHT / 2 - 0.2), Location.LEFT))
strains = [strain]

wave = Wave(grid)

fig, ax = plt.subplots()
fig_2, ax_2 = utils.create_visualized_subplots(wave, obstacles, strains)
fig_3, ax_3 = plt.subplots()

ims = []
times = np.arange(0, OBSERVATION_TIME + DT, DT)

observation_values: list[float] = []

for time in times:
    wave.update(obstacles, strains)
    im = ax.imshow(
        wave.get_values().T,
        "binary",
        vmin=-0.01,
        vmax=0.01,
        extent=[0, WIDTH, 0, HEIGHT],
        origin="lower",
    )
    title = ax.text(
        0.5,
        1.01,
        f"Time = {round(time,2)}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize="large",
    )
    ims.append([im, title])

ax.invert_yaxis()
for obstacle in obstacles:
    ax.plot(obstacle.xs(), obstacle.ys(), color="k")

fig_2.legend(ncol=6, loc="upper center")
ax_3.plot(times, [strain.input(0.0, 0.0, t) for t in times], color="k")
ax_3.set(title="input", xlabel="Time")
anim = ArtistAnimation(fig, ims, interval=10)

plt.show()

# output_dir_name = "2022_result"
# os.makedirs(output_dir_name, exist_ok=True)
# anim.save(os.path.join(output_dir_name, "2022_issue.gif"))
# fig_2.savefig(os.path.join(output_dir_name, "grid.png"))
# fig_3.savefig(os.path.join(output_dir_name, "input.png"))
