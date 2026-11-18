"""Boundary setup for car_les: y flow direction, z vertical, moving ground.

Matches car_les/0/U, 0/p and 0/include/*. No periodic boundaries.
Coordinates are midpoints of exposed VTK polygon edges at each snapshot time.
"""

import numpy as np
import torch

from cfd_pinn.physics.boundary_conditions import Boundary, BoundaryConditions, FunctionCondition, ValueCondition


def create_car_boundaries(snapshots, velocity: float ) -> BoundaryConditions:
    points = {boundary: [] for boundary in Boundary}
    car_normals = []
    for time, data in snapshots:
        segments = data.boundary_segments()[:, :, [1, 2]]
        midpoints = segments.mean(axis=1)
        lower = data.points[:, [1, 2]].min(axis=0)
        upper = data.points[:, [1, 2]].max(axis=0)
        # Both endpoints must lie on an outer side; all remaining edges are Car.
        masks = {
            Boundary.LEFT: np.isclose(segments[:, :, 0], lower[0], atol=1e-7, rtol=0).all(axis=1),
            Boundary.RIGHT: np.isclose(segments[:, :, 0], upper[0], atol=1e-7, rtol=0).all(axis=1),
            Boundary.BOTTOM: np.isclose(segments[:, :, 1], lower[1], atol=1e-7, rtol=0).all(axis=1),
            Boundary.TOP: np.isclose(segments[:, :, 1], upper[1], atol=1e-7, rtol=0).all(axis=1),
        }
        masks[Boundary.CAR] = ~np.logical_or.reduce(list(masks.values()))
        for boundary, mask in masks.items():
            if not mask.any():
                raise ValueError(f"No {boundary.name} boundary segments at time {time}")
            coords = np.column_stack((midpoints[mask], np.full(mask.sum(), time)))
            points[boundary].append(torch.tensor(coords, dtype=torch.float32))
        tangent = segments[masks[Boundary.CAR], 1] - segments[masks[Boundary.CAR], 0]
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(lengths == 0):
            raise ValueError("Degenerate edge on the car surface")
        car_normals.append(torch.tensor(normals / lengths, dtype=torch.float32))

    if not car_normals:
        raise ValueError("No snapshots available for boundary conditions")
    points = {boundary: torch.cat(parts) for boundary, parts in points.items()}
    normals = torch.cat(car_normals)
    conditions = BoundaryConditions(fields=("uy", "uz", "p"), axes=("y", "z", "t"))

    def value(boundary, field, target):
        conditions.add(ValueCondition, boundary=boundary, coordinates=points[boundary],
                       field=field, value=target)

    def function(boundary, residual):
        conditions.add(FunctionCondition, boundary=boundary, coordinates=points[boundary],
                       function=residual)

    # Inlet and moving ground: U=(0, 2, 0), zero normal pressure gradient.
    for boundary, axis in ((Boundary.LEFT, "y"), (Boundary.BOTTOM, "z")):
        value(boundary, "uy", velocity)
        value(boundary, "uz", 0.0)
        function(boundary, lambda prediction, d, axis=axis: d.grad("p", [axis]))

    # Top slip: no normal flow, zero normal gradient of tangential velocity/p.
    value(Boundary.TOP, "uz", 0.0)
    function(Boundary.TOP, lambda prediction, d: d.grad("uy", ["z"]))
    function(Boundary.TOP, lambda prediction, d: d.grad("p", ["z"]))

    # Outlet p=0. inletOutlet velocity: zero gradient for outflow, U=0 for inflow.
    value(Boundary.RIGHT, "p", 0.0)
    for column, field in enumerate(("uy", "uz")):
        function(Boundary.RIGHT, lambda prediction, d, column=column, field=field: torch.where(
            prediction[:, 0] < 0, prediction[:, column], d.grad(field, ["y"])
        ))

    # Stationary car: no slip and zero pressure derivative along the local normal.
    value(Boundary.CAR, "uy", 0.0)
    value(Boundary.CAR, "uz", 0.0)

    def car_pressure_residual(prediction, derivatives):
        nonlocal normals
        normals = normals.to(prediction)

        return (
                normals[:, 0] * derivatives.grad("p", ["y"])
                + normals[:, 1] * derivatives.grad("p", ["z"])
        )

    function(Boundary.CAR, car_pressure_residual)
    return conditions
