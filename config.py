from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "samples"

CASES = {
    "car_les_2": 2.0,
    "car_les_3": 3.0,
    "car_les_4": 4.0,
    "car_les_5": 5.0,
}

N_HIDDEN = 64
N_HIDDEN_LAYERS = 3

EPOCHS = 100
LEARNING_RATE = 1e-3

CONTINUITY_WEIGHT = 1e-4
MOMENTUM_WEIGHT = 1e-6
BOUNDARY_WEIGHT = 1.0

SIMULATIONS_PER_BATCH = 1