from collections.abc import Sequence

import torch

from cfd_pinn.pinn import PINN
from cfd_pinn.data.batches import SimulationBatch


def train(
    network: PINN,
    optimizer: torch.optim.Optimizer,
    *,
    batches: Sequence[SimulationBatch],
    epochs: int,
    continuity_weight: float,
    momentum_weight: float,
    boundary_weight: float = 1.0,
):
    """Train prepared batches with equal simulation and frame weights.

    Update the model once per epoch. Batches must be on the model's device.
    The network and physics must process coordinate rows independently.
    """
    if not batches:
        raise ValueError("At least one batch is required")
    simulation_ids = [sim_id for batch in batches for sim_id in batch.simulation_ids]
    n_simulations = len(simulation_ids)

    loss_names = ("data", "physics", "continuity", "momentum", "boundary", "total")
    history = {
        "total": [],
        "samples": {
            sim_id: {name: [] for name in loss_names}
            for sim_id in simulation_ids
        }
    }
    network.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        epoch_total = 0.0

        sums_by_sim = {
            sim_id: {name: 0.0 for name in loss_names} for sim_id in simulation_ids
        }
        for batch in batches:
            batch_size = len(batch.simulation_ids)
            n_frames = len(batch.coords)
            n_points = batch.points_per_frame
            weight = batch_size / (n_simulations * n_frames)

            for frame_index in range(n_frames):

                coords = batch.coords[frame_index]
                coords = coords.detach().requires_grad_(True)
                targets = batch.targets[frame_index]
                nut = batch.nut[frame_index]

                prediction = network(coords)
                physical_prediction = network.normalizer.denormalize("fields", prediction)
                continuity, momentum_y, momentum_z = network.physics.physics_residuals(
                    coords, physical_prediction, nut
                )

                data_loss = (prediction - targets).square().reshape(batch_size, -1).mean(dim=1)
                continuity_loss = continuity.square().reshape(batch_size, n_points).mean(dim=1)
                momentum_loss = ((momentum_y.square() + momentum_z.square()).reshape(batch_size, n_points).mean(dim=1) / 2)

                physics_loss = (
                    continuity_weight * continuity_loss
                    + momentum_weight * momentum_loss
                )

                boundary_loss = torch.zeros_like(data_loss)
                if boundary_weight != 0:
                    boundary_loss = torch.stack([
                        boundaries.loss(network) for boundaries in batch.boundaries[frame_index]
                    ])

                total_loss = data_loss + physics_loss + boundary_weight * boundary_loss

                (total_loss.mean() * weight).backward()

                losses = {
                    "data": data_loss,
                    "physics": physics_loss,
                    "continuity": continuity_loss,
                    "momentum": momentum_loss,
                    "boundary": boundary_loss,
                    "total": total_loss,
                }

                for name, loss in losses.items():
                    for sim_id, value in zip(batch.simulation_ids, loss.detach()):
                        sums_by_sim[sim_id][name] += value / n_frames

        epoch_metrics = torch.stack([
            torch.stack([sums_by_sim[sim_id][name] for name in loss_names])
            for sim_id in simulation_ids
        ]).cpu().tolist()

        for sim_id, metrics in zip(simulation_ids, epoch_metrics):
            for name, value in zip(loss_names, metrics):
                history["samples"][sim_id][name].append(value)
            epoch_total += history["samples"][sim_id]["total"][-1] / n_simulations

        optimizer.step()
        history["total"].append(epoch_total)
        print(f"[Epoch {epoch:04d}] Mean total={epoch_total:.3e}", flush=True)

    return history
