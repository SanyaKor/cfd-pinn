import VTKData
from pathlib import Path
import torch
from normalizer import Normalizer, Mode
from pinn import PINN
from train import train


EPOCHS = 2000
CONTINUITY_WEIGHT = 1e-4
MOMENTUM_WEIGHT = 1e-6

if __name__ == "__main__":

    ROOT = Path.cwd()
    if not (ROOT / 'car_rans_2d').exists():
        ROOT = ROOT.parent

    CASE = ROOT / 'car_rans_2d'

    VTK_PATH = ROOT / 'carCentrePlane.vtk'
    # VTK_PATH = CASE / 'postProcessing/cutPlane/500/carCentrePlane.vtk'

    data = VTKData.VTKData(VTK_PATH)
    data.read()

    U = data.fields['U']

    ##################################

    yz_numpy = data.points[:, [1, 2]]
    velocity_numpy = U[:, [1, 2]]
    pressure_numpy = data.fields['p'][:, [0]]
    nut_numpy = data.fields['nut'][:, 0]

    ##################################

    yz = torch.tensor(yz_numpy, dtype=torch.float32)
    velocity = torch.tensor(velocity_numpy, dtype=torch.float32)
    pressure = torch.tensor(pressure_numpy, dtype=torch.float32)
    nut = torch.tensor(nut_numpy, dtype=torch.float32)

    ##################################

    FIELD_NAMES = ('Uy', 'Uz', 'p')
    targets_physical = torch.cat([velocity, pressure], dim=1)

    normalizer = Normalizer()
    yz_normalized = normalizer.normalize('coordinates', yz, Mode.MIN_MAX)
    targets_normalized = normalizer.normalize('fields', targets_physical, Mode.MAX_ABS)

    network = PINN(n_input=2,
                   n_output=3,
                   n_hidden=64,
                   n_hidden_layers=3,
                   normalizer=normalizer)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    train(network=network,
          optimizer=optimizer,
          continuity_weight=CONTINUITY_WEIGHT,
          momentum_weight=MOMENTUM_WEIGHT,
          epochs=EPOCHS,
          yz=yz,
          targets_normalized=targets_normalized,
          nut=nut)
