"""Boundary conditions with explicit value and derivative calculations."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import torch

from cfd_pinn.physics.cache_grad import CacheGrad

#TODO
# NEW IDEA
# boundaries.add(
#     ValueCondition,
#     boundary=Boundary.LEFT,
#     coordinates=left_points,
#     values={"uy": inlet_velocity, "uz": 0.0},
# )

class Boundary(Enum):
    TOP = 1
    BOTTOM = 2
    LEFT = 3
    RIGHT = 4
    CAR = 5


@dataclass(kw_only=True)
class BoundaryCondition(ABC):
    fields: tuple[str, ...]  # Network output columns.
    axes: tuple[str, ...]    # Coordinate columns.
    boundary: Boundary
    coordinates: torch.Tensor

    def __post_init__(self):
        for names in (self.fields, self.axes):
            if not names or any(not isinstance(name, str) or not name for name in names):
                raise ValueError("Fields and axes must contain nonempty names")
            if len(set(names)) != len(names):
                raise ValueError("Fields and axes must contain unique names")

    @abstractmethod
    def residual(self, network: torch.nn.Module) -> torch.Tensor:
        """Compute one residual row per boundary point."""

    def loss(self, network: torch.nn.Module) -> torch.Tensor:

        residual = self.residual(network)

        if not isinstance(residual, torch.Tensor) or residual.ndim == 0 or residual.numel() == 0:
            raise ValueError("Residual must be a nonempty tensor with one row per point")

        if residual.shape[0] != len(self.coordinates):
            raise ValueError("Residual must have one row per point")

        return residual.square().mean()

    def _predict(self, network, coordinates):

        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.axes) or len(coordinates) == 0:
            raise ValueError(f"Expected nonempty coordinates with columns {self.axes}")

        prediction = network.predict_physical(coordinates)

        if prediction.shape != (len(coordinates), len(self.fields)):
            raise ValueError(f"Expected network output with columns {self.fields}")

        return prediction

    def _derivatives(self, coordinates, prediction):
        derivatives = CacheGrad(fields=self.fields, axes=self.axes)
        derivatives.bind(coordinates, prediction)
        return derivatives


@dataclass(kw_only=True)
class ValueCondition(BoundaryCondition):
    field: str
    value: float

    def residual(self, network: torch.nn.Module) -> torch.Tensor:
        prediction = self._predict(network, self.coordinates)
        return prediction[:, self.fields.index(self.field)] - self.value


@dataclass(kw_only=True)
class FunctionCondition(BoundaryCondition):
    # Return residuals, not target values. Read values from prediction directly.
    function: Callable[[torch.Tensor, CacheGrad], torch.Tensor]

    def residual(self, network: torch.nn.Module) -> torch.Tensor:
        coordinates = self.coordinates.detach().clone().requires_grad_(True)
        prediction = self._predict(network, coordinates)
        derivatives = self._derivatives(coordinates, prediction)
        return self.function(prediction, derivatives)


@dataclass(kw_only=True)
class PeriodicCondition(BoundaryCondition):
    """Match row-aligned points; the caller supplies the pairing.

    derivative_orders=((),) matches values only; ((), ("x",)) also matches
    first x derivatives. Loss averages over points, fields and orders.
    """

    paired_boundary: Boundary
    paired_coordinates: torch.Tensor
    periodic_fields: tuple[str, ...]
    derivative_orders: tuple[tuple[str, ...], ...] = ((),)

    def __post_init__(self):
        super().__post_init__()

        if not self.periodic_fields or not self.derivative_orders:
            raise ValueError("Select at least one periodic field and derivative order")

        if self.coordinates.shape != self.paired_coordinates.shape:
            raise ValueError("Periodic coordinates must have matching shapes and paired rows")

    def residual(self, network: torch.nn.Module) -> torch.Tensor:

        left, right = self.coordinates, self.paired_coordinates

        if any(self.derivative_orders):
            left = left.detach().clone().requires_grad_(True)
            right = right.detach().clone().requires_grad_(True)

        left_prediction = self._predict(network, left)
        right_prediction = self._predict(network, right)

        if any(self.derivative_orders):
            left_derivatives = self._derivatives(left, left_prediction)
            right_derivatives = self._derivatives(right, right_prediction)

        residuals = []

        for field in self.periodic_fields:
            column = self.fields.index(field)
            for order in self.derivative_orders:
                if order:
                    residual = left_derivatives.grad(field, order) - right_derivatives.grad(field, order)
                else:
                    residual = left_prediction[:, column] - right_prediction[:, column]
                residuals.append(residual)

        return torch.stack(residuals, dim=1)


def compute_boundary_loss(network: torch.nn.Module, conditions: Sequence[BoundaryCondition]) -> torch.Tensor:
    """Sum independent condition losses; each condition evaluates the network."""

    if not conditions:
        raise ValueError("At least one boundary condition is required")

    return torch.stack([condition.loss(network) for condition in conditions]).sum()


@dataclass(kw_only=True)
class BoundaryConditions:
    """Create and collect conditions with a shared field and axis layout."""

    fields: tuple[str, ...]
    axes: tuple[str, ...]
    conditions: list[BoundaryCondition] = field(default_factory=list, init=False)

    def add(self, condition_type: type[BoundaryCondition], **kwargs) -> BoundaryCondition:
        condition = condition_type(fields=self.fields, axes=self.axes, **kwargs)
        self.conditions.append(condition)
        return condition

    def add_many(self, conditions: Sequence[tuple[type[BoundaryCondition], dict]]) -> list[BoundaryCondition]:
        """Append (condition class, parameters) pairs with the shared layout."""
        created = [
            condition_type(fields=self.fields, axes=self.axes, **parameters)
            for condition_type, parameters in conditions
        ]
        self.conditions.extend(created)
        return created

    def loss(self, network: torch.nn.Module) -> torch.Tensor:
        return compute_boundary_loss(network, self.conditions)
