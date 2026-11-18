from pathlib import Path
from collections.abc import Mapping

import numpy as np

from cfd_pinn.data.VTKData import VTKData
from cfd_pinn.data.simulations import Simulation
from cfd_pinn.physics.car_boundaries import create_car_boundaries


def load_simulations(data_root: Path, cases: Mapping[str, float]) -> list[Simulation]:
    simulations = []

    for case_name, velocity in cases.items():
        case_path = data_root / case_name / "postProcessing" / "cutPlane"

        if not case_path.is_dir():
            raise FileNotFoundError(f"No such folder: {case_path}")

        snapshots = VTKData.read_time_series(case_path)

        coords = []
        fields = []
        nut = []
        boundaries = []

        for time, data in snapshots:
            time_column = np.full((len(data.points), 1), time)

            coords.append(np.column_stack((data.points[:, [1, 2]], time_column)))
            fields.append(
                np.column_stack((
                    data.fields["U"][:, [1, 2]],
                    data.fields["p"][:, 0],
                ))
            )
            nut.append(data.fields["nut"][:, 0])

            boundaries.append(create_car_boundaries([(time, data)], velocity=velocity))

        if not coords:
            raise ValueError(f"No snapshots found : {case_path}")

        simulations.append(
            Simulation(
                axes=("y", "z", "t"),
                parameters={"velocity": velocity},
                coords=coords,
                fields=fields,
                nut=nut,
                boundaries=boundaries
            )
        )

    return simulations