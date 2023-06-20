from __future__ import annotations
import matplotlib.pyplot as plt
from .domains import *


def create_visualized_subplots(
    wave: Wave, obstacles: list[Obstacle] | None = None, strains: list[Strain] | None = None
):
    corner_color = ["purple"]
    color_map = {location: color for location, color in zip(Location, ["r", "b", "y", "g"] + corner_color * 4)}
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    indices_map = wave.grid.calculate_boundary_indices()

    for location, indices in indices_map.items():
        X, Y = indices
        if len(X):
            ax.scatter(X, Y, color=color_map[location])

    if obstacles is None:
        return fig, ax

    for obstacle in obstacles:
        indices_map = wave.grid.calculate_obstacle_indices(obstacle)
        for location, indices in indices_map.items():
            X, Y = indices
            if len(X):
                ax.scatter(X, Y, color=color_map[location])

    if strains is None:
        return fig, ax

    for strain in strains:
        indices_map = wave.grid.calculate_strain_indices(strain)
        for location, indices in indices_map.items():
            X, Y = indices
            if len(X):
                ax.scatter(X, Y, color="k")

    return fig, ax
