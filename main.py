import VTKData
from pathlib import Path


if __name__ == "__main__":

    ROOT = Path.cwd()
    if not (ROOT / 'car_rans_2d').exists():
        ROOT = ROOT.parent

    CASE = ROOT / 'car_rans_2d'

    VTK_PATH = ROOT / 'carCentrePlane.vtk'
    # VTK_PATH = CASE / 'postProcessing/cutPlane/500/carCentrePlane.vtk'

    data = VTKData.VTKData(VTK_PATH)
    data.read()

    print(data.fields.keys())