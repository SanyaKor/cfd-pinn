from pathlib import Path
from collections import Counter
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
        self.polygons = []

    def read(self):
        tokens = self.path.read_text().split()

        ##################
        # Points Geometry - location
        ##################
        points_index = tokens.index('POINTS')
        points_size = int(tokens[points_index + 1])

        points_geometry = np.asarray(tokens[points_index + 3:points_index + 3 + 3 * points_size],dtype=np.float64)
        points_geometry = points_geometry.reshape(points_size, 3)

        polygon_index = tokens.index('POLYGONS')
        polygon_count = int(tokens[polygon_index + 1])
        ptr = polygon_index + 3
        self.polygons = []
        for _ in range(polygon_count):
            size = int(tokens[ptr])
            self.polygons.append(tuple(map(int, tokens[ptr + 1:ptr + 1 + size])))
            ptr += size + 1


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

    def boundary_segments(self):
        """Return endpoint coordinates of edges belonging to just one polygon."""
        edges = Counter()
        for polygon in self.polygons:
            edges.update(
                tuple(sorted((a, b)))
                for a, b in zip(polygon, polygon[1:] + polygon[:1])
            )
        indices = np.asarray([edge for edge, count in edges.items() if count == 1], dtype=int)
        if indices.size == 0:
            raise ValueError(f"No boundary edges found in {self.path}")
        return self.points[indices]

    @staticmethod
    def read_time_series(folder: Path):
        snapshots = []

        for directory in folder.iterdir():
            if not directory.is_dir():
                continue

            try:
                time = float(directory.name)
            except ValueError:
                continue

            data = VTKData(directory / "carCentrePlane.vtk")
            data.read()

            snapshots.append((time, data))

        snapshots.sort(key=lambda snapshot: snapshot[0])
        return snapshots












