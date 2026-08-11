import torch
from normalizer import Normalizer

class PINN(torch.nn.Module):

    def __init__(self, n_input : int, n_hidden : int, n_output : int, n_hidden_layers : int, normalizer : Normalizer):

        super().__init__()
        self.normalizer = normalizer

        activation_function = torch.nn.Tanh

        layers = [torch.nn.Linear(n_input, n_hidden), activation_function()]

        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(n_hidden, n_hidden), activation_function()])

        layers.append(torch.nn.Linear(n_hidden, n_output))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, yz_physical : torch.Tensor) -> torch.Tensor:
        yz_normalized = self.normalizer.transform('coordinates', yz_physical)
        return self.network(yz_normalized)

    def predict_physical(self, yz_physical : torch.Tensor) -> torch.Tensor:
        prediction_normalized = self.forward(yz_physical)
        return self.normalizer.denormalize('fields', prediction_normalized)


def gradient(value : torch.Tensor, coordinates : torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value, coordinates,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]


def physics_residuals(network : PINN, coordinates_physical : torch.Tensor, nut_values):

    coordinates_physical = (coordinates_physical.detach().clone().requires_grad_(True))
    prediction_physical = network.predict_physical(coordinates_physical)

    uy = prediction_physical[:, 0]
    uz = prediction_physical[:, 1]
    p = prediction_physical[:, 2]

    grad_uy = gradient(uy, coordinates_physical)
    grad_uz = gradient(uz, coordinates_physical)
    grad_p = gradient(p, coordinates_physical)

    uy_y = grad_uy[:, 0]
    uy_z = grad_uy[:, 1]

    uz_y = grad_uz[:, 0]
    uz_z = grad_uz[:, 1]

    p_y = grad_p[:, 0]
    p_z = grad_p[:, 1]

    uy_yy = gradient(uy_y, coordinates_physical)[:, 0]
    uy_zz = gradient(uy_z, coordinates_physical)[:, 1]
    uz_yy = gradient(uz_y, coordinates_physical)[:, 0]
    uz_zz = gradient(uz_z, coordinates_physical)[:, 1]

    continuity = uy_y + uz_z

    #### NU = 1.5e-5  check out physical properties in constant folder
    nu_effective = 1.5e-5 + nut_values

    momentum_y = uy * uy_y + uz * uy_z + p_y - nu_effective * (uy_yy + uy_zz)
    momentum_z = uy * uz_y + uz * uz_z + p_z - nu_effective * (uz_yy + uz_zz)

    return continuity, momentum_y, momentum_z


