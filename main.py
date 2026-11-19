import torch
from cfd_pinn.normalizer import Normalizer, Mode
from cfd_pinn.pinn import PINN
from cfd_pinn.train import train

from cfd_pinn.physics.incompressible_les import IncompressibleLES
import config
from cfd_pinn.data.loader import load_simulations
from cfd_pinn.data.batches import prepare_batches


if __name__ == "__main__":

    simulations = load_simulations(config.DATA_ROOT, config.CASES)
    normalizer = Normalizer()

    normalizer.fit(
        "coordinates",
        torch.cat([coords for sim in simulations for coords in sim.coords]),
        Mode.MIN_MAX
    )

    normalizer.fit(
        "fields",
        torch.cat([fields for sim in simulations for fields in sim.fields]),
        Mode.MAX_ABS
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print(f"Device: {device}")

    for sim in simulations:
        sim.fields_normalized = [normalizer.transform("fields", fields) for fields in sim.fields]

    batches = prepare_batches(simulations, config.SIMULATIONS_PER_BATCH)

    normalizer.to(device)
    for batch in batches:
        batch.to(device)

    network = PINN(
        n_input=4,
        n_output=3,
        n_hidden=config.N_HIDDEN,
        n_hidden_layers=config.N_HIDDEN_LAYERS,
        normalizer=normalizer,
        physics=IncompressibleLES()
    ).to(device)


    optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE)

    history = train(
        network=network,
        optimizer=optimizer,
        batches=batches,
        epochs=config.EPOCHS,
        continuity_weight=config.CONTINUITY_WEIGHT,
        momentum_weight=config.MOMENTUM_WEIGHT,
        boundary_weight=config.BOUNDARY_WEIGHT
    )

    output_dir = config.ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = output_dir / "pinn_batched_7700.pt"

    torch.save(
        {
            "model": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.parameters,
            "history": history,
            "model_config": {
                "n_input": 4,
                "n_output": 3,
                "n_hidden": config.N_HIDDEN,
                "n_hidden_layers": config.N_HIDDEN_LAYERS,
            },
            "training_config": {
                "epochs": config.EPOCHS,
                "learning_rate": config.LEARNING_RATE,
                "simulations_per_batch": config.SIMULATIONS_PER_BATCH,
                "continuity_weight": config.CONTINUITY_WEIGHT,
                "momentum_weight": config.MOMENTUM_WEIGHT,
                "boundary_weight": config.BOUNDARY_WEIGHT,
            },
            "training_cases": dict(config.CASES),
            "epochs": len(history["total"]),
            "simulation_velocities": {
                sim.id: sim.parameters["velocity"]
                for sim in simulations
            },
        },
        checkpoint_path
    )

    print(f"Saved checkpoint: {checkpoint_path}")