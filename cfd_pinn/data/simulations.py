from typing import Sequence
import numpy as np
from itertools import count
from cfd_pinn.physics.boundary_conditions import PeriodicCondition

import torch
from cfd_pinn.physics.boundary_conditions import BoundaryConditions

class Simulation:

    _id_counter = count(1)

    def __init__(
        self,
        *,
        axes: tuple[str, ...],
        parameters: dict[str, float],
        coords: Sequence[np.ndarray],
        fields: Sequence[np.ndarray],
        nut: Sequence[np.ndarray],
        boundaries: list[BoundaryConditions]
    ):
        if len(coords) == 0:
            raise ValueError("Simulation must contain at least one snapshot")
        if not (len(coords) == len(fields) == len(nut) == len(boundaries)):
            raise ValueError("coords, fields, nut and boundaries must have the same number of snapshots")

        self.axes = axes
        self.parameters = dict(sorted(parameters.items()))

        self.fields = [torch.as_tensor(frame, dtype=torch.float32) for frame in fields]
        self.coords = [torch.as_tensor(frame, dtype=torch.float32) for frame in coords]
        self.nut = [torch.as_tensor(frame, dtype=torch.float32) for frame in nut]

        self.boundaries = boundaries

        self.append_parameters()
        self.append_boundary_parameters()
        self.id = next(self._id_counter)

    def append_parameters(self) -> None:
        n_axes = len(self.axes)
        n_inputs = n_axes + len(self.parameters)
        sizes = [coords.shape[1] for coords in self.coords]

        if all(size == n_inputs for size in sizes):
            return

        if any(size != n_axes for size in sizes):
            raise ValueError("Parameters are partially added or coordinates have invalid columns")

        values = list(self.parameters.values())

        for i, coords in enumerate(self.coords):
            params = coords.new_tensor(values).expand(len(coords), -1)
            self.coords[i] = torch.cat((coords, params), dim=1)

    def append_boundary_parameters(self) -> None:
        values = list(self.parameters.values())

        n_axes = len(self.axes)
        n_inputs = n_axes + len(values)

        for boundaries in self.boundaries:
            for condition in boundaries.conditions:
                names = ["coordinates"]

                if isinstance(condition, PeriodicCondition):
                    names.append("paired_coordinates")

                for name in names:
                    coords = getattr(condition, name)

                    if coords.shape[1] == n_inputs:
                        continue

                    if coords.shape[1] != n_axes:
                        raise ValueError("Invalid boundary coordinate columns")

                    params = coords.new_tensor(values).expand(len(coords), -1)
                    setattr(condition, name, torch.cat((coords, params), dim=1))

                condition.axes = self.axes + tuple(self.parameters)

            boundaries.axes = self.axes + tuple(self.parameters)

    def to(self, device) -> None:
        self.coords = [tensor.to(device) for tensor in self.coords]
        self.fields = [tensor.to(device) for tensor in self.fields]
        self.fields_normalized = [tensor.to(device) for tensor in self.fields_normalized]
        self.nut = [tensor.to(device) for tensor in self.nut]

        for boundaries in self.boundaries:
            for condition in boundaries.conditions:
                condition.coordinates = condition.coordinates.to(device)

                if isinstance(condition, PeriodicCondition):
                    condition.paired_coordinates = (condition.paired_coordinates.to(device))
