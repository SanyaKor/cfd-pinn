# CFD PINN

A small educational Physics-Informed Neural Network for approximating a steady OpenFOAM RANS solution.

Repository: [github.com/SanyaKor/openfoam-car-aerodynamics](https://github.com/SanyaKor/openfoam-car-aerodynamics)

The project was originally based on a 3D automotive RANS case, which was adapted to a 2D OpenFOAM case. The OpenFOAM `cutPlane` function object was then used to extract the coordinates and flow fields from the computational plane and export them to VTK.

The network maps a point in the 2D domain to velocity and pressure:

```text
(y, z) -> (Uy, Uz, p)
```

## Running

Install the dependencies:

```bash
uv sync
```

Open `testing.ipynb`, select `.venv/bin/python` as the interpreter, and run all cells from top to bottom.

Training can also be started without the notebook:

```bash
uv run python main.py
```

## Data

The training data is loaded from `carCentrePlane.vtk`:

- `y`, `z` — point coordinates;
- `Uy`, `Uz` — velocity components;
- `p` — pressure;
- `nut` — turbulent kinematic viscosity.

The coordinates are normalized to `[-1, 1]`. Each output field is divided by its own maximum absolute value.

## Model

The PINN has three hidden layers with 64 neurons per layer and `Tanh` activation functions.

The total training loss is:

```text
total = data_loss
      + continuity_weight * continuity_loss
      + momentum_weight * momentum_loss
```

- `data` measures the difference between the PINN prediction and OpenFOAM data;
- `continuity` is the incompressible continuity-equation residual;
- `momentum` is the steady momentum-equation residual.

The current weights are:

```python
CONTINUITY_WEIGHT = 1e-4
MOMENTUM_WEIGHT = 1e-6
```

The training log prints raw physics losses. For example, `momentum=587` contributes only `587 * 1e-6 = 0.000587` to the total loss.

`nut` is currently taken directly from the OpenFOAM solution. The viscous momentum term uses the simplified form `nu_effective * Laplacian(U)`.

## Output

The notebook displays:

- training convergence;
- OpenFOAM fields;
- PINN predictions;
- absolute errors;
- RMSE for `Uy`, `Uz`, and `p`.

## Project files

```text
VTKData.py       VTK reader
normalizer.py    data normalization
pinn.py          PINN and physics residuals
train.py         training loop
main.py          Python entry point
testing.ipynb    training and visualization
```