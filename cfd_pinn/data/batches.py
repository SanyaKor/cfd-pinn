from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch

from cfd_pinn.data.simulations import Simulation
from cfd_pinn.physics.boundary_conditions import BoundaryConditions, PeriodicCondition


@dataclass
class SimulationBatch:
    simulation_ids: list[int]
    points_per_frame: int
    coords: list[torch.Tensor]
    targets: list[torch.Tensor]
    nut: list[torch.Tensor]
    boundaries: list[list[BoundaryConditions]]

    def to(self, device) -> None:
        self.coords = [tensor.to(device) for tensor in self.coords]
        self.targets = [tensor.to(device) for tensor in self.targets]
        self.nut = [tensor.to(device) for tensor in self.nut]

        for frame_boundaries in self.boundaries:
            for boundaries in frame_boundaries:
                for condition in boundaries.conditions:

                    condition.coordinates = condition.coordinates.to(device)
                    if isinstance(condition, PeriodicCondition):
                        condition.paired_coordinates = condition.paired_coordinates.to(device)


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
        # Device transfer must not modify the source simulation boundaries.
        batch.boundaries.append(deepcopy(boundaries))

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

