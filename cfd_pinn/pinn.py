import torch
from .normalizer import Normalizer
from cfd_pinn.physics.base import PhysicsSystem

class PINN(torch.nn.Module):
    def __init__(self, n_input : int, n_hidden : int, n_output : int, n_hidden_layers : int, normalizer : Normalizer, physics : PhysicsSystem):

        super().__init__()
        self.normalizer = normalizer
        self.physics = physics

        activation_function = torch.nn.Tanh

        layers = [torch.nn.Linear(n_input, n_hidden), activation_function()]

        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(n_hidden, n_hidden), activation_function()])

        layers.append(torch.nn.Linear(n_hidden, n_output))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, coords_physical : torch.Tensor) -> torch.Tensor:
        coords_normalized = self.normalizer.transform('coordinates', coords_physical)
        return self.network(coords_normalized)

    def predict_physical(self, coords_physical : torch.Tensor) -> torch.Tensor:
        prediction_normalized = self.forward(coords_physical)
        return self.normalizer.denormalize('fields', prediction_normalized)



