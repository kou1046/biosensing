from __future__ import annotations
import matplotlib.pyplot as plt
from . import domain

corner_color = ["purple"]
corner_label = ["corner"]
color_map = {location: color for location, color in zip(domain.Location, ["r", "b", "y", "g"] + corner_color * 4)}
label_map = {
    location: label
    for location, label in zip(domain.Location, ["right", "left", "top", "bottom"] + corner_label + [None, None, None])
}


def create_visualized_subplots(
    wave: domain.Wave, obstacles: list[domain.Obstacle] | None = None, strains: list[domain.Strain] | None = None
):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_aspect("equal")

    indices_map = wave.grid.calculate_boundary_indices()

    if obstacles is not None:
        for obstacle in obstacles:
            obstacle_indices = wave.grid.calculate_obstacle_indices(obstacle)
            for location in domain.Location:
                X, Y = indices_map[location]
                wall_X, wall_Y = obstacle_indices[location]
                X.extend(wall_X)
                Y.extend(wall_Y)

    for location, indices in indices_map.items():
        ax.scatter(*indices, color=color_map[location], label=label_map[location])

    if strains is None:
        return fig, ax

    is_labeling = False
    for strain in strains:
        strain_indices_map = wave.grid.calculate_strain_indices(strain)
        for location, indices in strain_indices_map.items():
            X, Y = indices
            if len(X):
                ax.scatter(X, Y, color="k", label="input" if not is_labeling else None)
                is_labeling = True

    return fig, ax
