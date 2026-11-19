from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch

from cfd_pinn.data.simulations import Simulation
from cfd_pinn.physics.boundary_conditions import (
    Boundary, BoundaryBatch, BoundaryConditions, BoundaryGroup, BoundaryPart,
    FunctionCondition, ValueCondition,
)


@dataclass
class SimulationBatch:
    simulation_ids: list[int]
    points_per_frame: int
    coords: list[torch.Tensor]
    targets: list[torch.Tensor]
    nut: list[torch.Tensor]
    boundaries: list[BoundaryBatch]

    def to(self, device) -> None:
        self.coords = [tensor.to(device) for tensor in self.coords]
        self.targets = [tensor.to(device) for tensor in self.targets]
        self.nut = [tensor.to(device) for tensor in self.nut]

        for frame in self.boundaries:
            frame.to(device)


def prepare_batches(simulations: Sequence[Simulation], simulations_per_batch: int = 1) -> list[SimulationBatch]:

    if not simulations:
        raise ValueError("At least one simulation is required")

    if (
        isinstance(simulations_per_batch, bool)
        or not isinstance(simulations_per_batch, int)
        or simulations_per_batch < 1
    ):
        raise ValueError("simulations_per_batch must be a positive integer")

    _validate_simulations(simulations)

    batches = []
    for start in range(0, len(simulations), simulations_per_batch):
        group = simulations[start:start + simulations_per_batch]
        batches.append(_combine_simulations(group))
    return batches


def _combine_simulations(group: Sequence[Simulation]) -> SimulationBatch:

    n_frames = len(group[0].coords)
    batch = SimulationBatch(
        simulation_ids=[sim.id for sim in group],
        points_per_frame=len(group[0].coords[0]),
        coords=[],
        targets=[],
        nut=[],
        boundaries=[],
    )

    for frame_index in range(n_frames):
        coords = []
        targets = []
        nut = []
        boundaries = []

        for sim in group:
            coords.append(sim.coords[frame_index])
            targets.append(sim.fields_normalized[frame_index])
            nut.append(sim.nut[frame_index])
            boundaries.append(sim.boundaries[frame_index])

        batch.coords.append(torch.cat(coords))
        batch.targets.append(torch.cat(targets))
        batch.nut.append(torch.cat(nut))
        batch.boundaries.append(_prepare_boundaries(boundaries))

    return batch


def _validate_simulations(simulations: Sequence[Simulation]) -> None:

    n_frames = len(simulations[0].coords)
    if n_frames == 0:
        raise ValueError("Simulations must contain frames")

    n_points = len(simulations[0].coords[0])

    if n_points == 0:
        raise ValueError("Frames must contain points")

    for sim in simulations:
        if any(len(values) != n_frames for values in (
            sim.coords, sim.fields_normalized, sim.nut, sim.boundaries,
        )):
            raise ValueError("All simulations must have the same number of frames")

        for values in (sim.coords, sim.fields_normalized, sim.nut):

            if any(len(tensor) != n_points for tensor in values):
                raise ValueError("All frames must have the same number of points")

            if any(tensor.device.type != "cpu" for tensor in values):
                raise ValueError("Prepare batches before transferring simulations to GPU")



def _prepare_boundaries(simulations: list[BoundaryConditions]) -> BoundaryBatch:
    """Build boundary tensors and row slices once, on CPU."""

    # Keep source simulation coordinates independent of device transfers.
    simulations = deepcopy(simulations)
    frame = BoundaryBatch(simulations=simulations, groups=[], separate_conditions=[])
    fields = simulations[0].fields
    axes = simulations[0].axes

    for simulation_index, boundaries in enumerate(simulations):
        if not boundaries.conditions:
            raise ValueError("Each simulation must have boundary conditions")
        if boundaries.fields != fields or boundaries.axes != axes:
            raise ValueError("Boundary field and axis layouts must match across simulations")
        for condition in boundaries.conditions:
            if condition.coordinates.device.type != "cpu":
                raise ValueError("Prepare boundaries before transferring them to GPU")
            if condition.fields != fields or condition.axes != axes:
                raise ValueError("Condition layout must match its boundary collection")
            if type(condition) not in (ValueCondition, FunctionCondition):
                frame.separate_conditions.append((simulation_index, condition))

    for boundary in Boundary:
        coordinates = []
        parts = []
        start = 0

        for simulation_index, boundaries in enumerate(simulations):
            conditions = [
                condition for condition in boundaries.conditions
                if condition.boundary == boundary
                and type(condition) in (ValueCondition, FunctionCondition)
            ]
            if not conditions:
                continue

            points = conditions[0].coordinates
            if points.ndim != 2 or points.shape[1] != len(axes) or len(points) == 0:
                raise ValueError("Boundary coordinates must be nonempty and match the axes")
            for condition in conditions[1:]:
                if not torch.equal(condition.coordinates, points):
                    raise ValueError("Conditions on the same boundary must use identical points")

            end = start + len(points)
            parts.append(BoundaryPart(simulation_index, slice(start, end), conditions))
            coordinates.append(points)
            start = end

        if coordinates:
            frame.groups.append(BoundaryGroup(torch.cat(coordinates), fields, axes, parts))

    return frame
