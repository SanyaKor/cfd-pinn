from collections.abc import Sequence
import torch

class CacheGrad:
    """Cached derivatives for independent rows of coordinates and predictions.

    Bind every new forward pass. Enable coordinate gradients before that pass
    and do not detach predictions. Results have shape (N,).
    """

    def __init__(self, fields: tuple[str, ...], axes: tuple[str, ...]):

        for label, names in (("fields", fields), ("axes", axes)):
            if not names or any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"{label} must contain nonempty string names")
            if len(set(names)) != len(names):
                raise ValueError(f"{label} must contain unique names")

        self.field_indices = {name: index for index, name in enumerate(fields)}
        self.coordinate_indices = {name: index for index, name in enumerate(axes)}
        self.coords = None
        self.prediction = None
        self.cache = {}
        self.grad_cache = {}

    def bind(self, coords: torch.Tensor, prediction: torch.Tensor):

        if not coords.requires_grad:
            raise ValueError("Enable coords.requires_grad before computing predictions")

        self.coords = coords
        self.prediction = prediction
        self.cache.clear()
        self.grad_cache.clear()

    def grad(self, field: str, wrt: Sequence[str] = ()) -> torch.Tensor:
        """Return the field, or a derivative such as grad('u', ['x', 'y'])."""

        if self.coords is None:
            raise RuntimeError("Call bind(coords, prediction) before grad")
        if field not in self.field_indices:
            raise ValueError(f"Unknown field: {field}")
        if isinstance(wrt, str):
            raise TypeError("Pass axes as a sequence, for example ['x'], not 'x'")

        axes = tuple(wrt)
        for axis in axes:
            if axis not in self.coordinate_indices:
                raise ValueError(f"Unknown axis: {axis}")

        key = (field, axes)

        if key in self.cache:
            return self.cache[key]

        if not axes:
            result = self.prediction[:, self.field_indices[field]]
            self.cache[key] = result
            return result

        previous_axes = axes[:-1]
        previous_value = self.grad(field, previous_axes)
        grad_key = (field, previous_axes)

        if grad_key not in self.grad_cache:
            gradient = None

            if previous_value.requires_grad:
                gradient = torch.autograd.grad(
                    previous_value,
                    self.coords,
                    grad_outputs=torch.ones_like(previous_value),
                    create_graph=True,
                    allow_unused=True
                )[0]

            if gradient is None:
                gradient = self.coords * 0.0 + previous_value.sum() * 0.0

            self.grad_cache[grad_key] = gradient

        result = self.grad_cache[grad_key][:, self.coordinate_indices[axes[-1]]]
        self.cache[key] = result
        return result
