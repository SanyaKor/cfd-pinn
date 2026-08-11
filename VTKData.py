from pathlib import Path
import numpy as np

class VTKData:

    SKIP_FIELDS = {
        "pointLevel",
        "cellLevel"
    }

    def __init__(self, path : Path):
        self.path = path
        self.points = None
        self.fields = None

    def read(self):
        tokens = self.path.read_text().split()

        ##################
        # Points Geometry - location
        ##################
        points_index = tokens.index('POINTS')
        points_size = int(tokens[points_index + 1])

        points_geometry = np.asarray(tokens[points_index + 3:points_index + 3 + 3 * points_size],dtype=np.float64)
        points_geometry = points_geometry.reshape(points_size, 3)


        ##################
        # Points Physics - physical properties of points
        ##################
        points_data_index = tokens.index('POINT_DATA')
        points_data_size = int(tokens[points_data_index + 4])

        ptr = points_data_index + 5
        fields = {}

        for i in range(points_data_size):
            field_name = tokens[ptr]

            # physical field dim - pdim
            # U -> (3) 7714 float

            # points_size
            # U 3 -> (7714) float

            pdim, points_size = int(tokens[ptr + 1]), int(tokens[ptr + 2])

            ptr += 4
            dim = pdim * points_size

            if field_name not in self.SKIP_FIELDS:
                fields[field_name] = np.asarray(tokens[ptr:ptr + dim], dtype=np.float64).reshape(points_size, pdim)

            ptr += dim

        self.points = points_geometry
        self.fields = fields














