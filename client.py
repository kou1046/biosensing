from __future__ import annotations
import matplotlib as mpl
from matplotlib.animation import ArtistAnimation
import numpy as np
import matplotlib.pyplot as plt

from wavesimulator import *


class SimpleStrain(Strain):
    def __init__(self, wall: Wall):
        super().__init__(wall)

    def input(self, x: float, y: float, t: float) -> float:
        f = 3
        return np.cos(2 * np.pi * f * t) if t <= 3.0 else 0.0


class GaussStrain(Strain):
    def __init__(self, wall: Wall):
        super().__init__(wall)

    def input(self, x: float, y: float, t: float) -> float:
        A = 3.0
        peak_time = 1.0
        sigma = 3.0
        return A * np.exp(-1 * (t - peak_time) ** 2 / 2 * sigma**2)


l = 3.0
width = 2.0 * l
height = l
h = 0.01
dt = 0.005
grid = Grid(width, height, h, dt)
pt1_1 = ((l / 2) - (l / 4), (l / 2) - (l / 4))
pt1_2 = ((l / 2) + (l / 4), (l / 2) - (l / 4))
pt1_3 = ((l / 2) + (l / 4), (l / 2) + (l / 4))
pt1_4 = ((l / 2) - (l / 4), (l / 2) + (l / 4))
walls = [
    Wall(pt1_1, pt1_2, Location.BOTTOM),
    Wall(pt1_2, pt1_3, Location.LEFT),
    Wall(pt1_3, pt1_4, Location.TOP),
    Wall(pt1_4, pt1_1, Location.RIGHT),
]
pt2_1 = (l, 0)
pt2_2 = (l, l / 2)
pt2_3 = (2.0 * l, l / 2)
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
wave = Wave(grid)
mpl.rcParams["animation.ffmpeg_path"] = "C:/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
mpl.rcParams["figure.subplot.hspace"] = 0.60
mpl.rcParams["figure.figsize"] = (20, 10)
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["font.size"] = 24
mpl.rcParams["axes.titleweight"] = "bold"
fig, axes = plt.subplots(3, 1)
axes[1].set(title="input")
axes[2].set(title="target_value")
ims = []
times = []
input_values = []
target_x, target_y = (width, 2 * l / 3)
target_values = []
for time in np.arange(0, 15.0 + dt, dt):
    wave.update(obstacles, strains)
    times.append(time)
    target_values.append(wave.get_value_by_cor((target_x, target_y)))
    im = axes[0].imshow(wave.values.T, "binary", vmin=-0.01, vmax=0.01)
    title = axes[0].text(
        0.5,
        1.01,
        f"Time = {round(time,2)}",
        ha="center",
        va="bottom",
        transform=axes[0].transAxes,
        fontsize="large",
    )
    input_values.append(strain_1.input(0, 0, time))
    im_2 = axes[1].plot(times, input_values, "k")
    im_3 = axes[2].plot(times, target_values, "k")
    ims.append([im, title] + im_2 + im_3)
anim = ArtistAnimation(fig, ims, interval=10, blit=True)
anim.save("sample.mp4", writer="ffmpeg")
