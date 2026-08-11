import torch
from pinn import PINN, physics_residuals

def train(network: PINN,
          optimizer: torch.optim.Optimizer,
          *,
          epochs : int,
          continuity_weight: float,
          momentum_weight: float,
          yz: torch.Tensor,
          nut: torch.Tensor,
          targets_normalized : torch.Tensor):

    history = {
        "data": [],
        "continuity": [],
        "momentum": [],
        "total": []
    }

    network.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        prediction = network(yz)
        data_loss = torch.mean((prediction - targets_normalized) ** 2)

        continuity, momentum_y, momentum_z = physics_residuals(network, yz, nut)
        continuity_loss = torch.mean(continuity ** 2)
        momentum_loss = torch.mean(momentum_y ** 2 + momentum_z ** 2) / 2

        total_loss = (data_loss
                      + continuity_weight * continuity_loss
                      + momentum_weight * momentum_loss)
        total_loss.backward()
        optimizer.step()

        history["data"].append(data_loss.item())
        history["continuity"].append(continuity_loss.item())
        history["momentum"].append(momentum_loss.item())
        history["total"].append(total_loss.item())

        if epoch == 1 or epoch % 100 == 0:
            print(
                f'{epoch:4d} | data={data_loss.item():.6f} '
                f'| total={total_loss.item():.6f}')

    return history





