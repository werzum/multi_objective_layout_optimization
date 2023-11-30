import numpy as np


class Point_3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    def distance(self, other):
        """Returns the distance between two 3dpoints"""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


class LineString_3D:
    def __init__(self, start_point: Point_3D, end_point: Point_3D):
        self.start_point = start_point
        self.end_point = end_point

    def interpolate(self, distance: float) -> Point_3D:
        """Returns the interpolated point at a given distance from the start point
        Args:
            distance (float): distance from the start point
        Returns:
            Point_3D: interpolated point"""

        vector = self.end_point.xyz - self.start_point.xyz
        # normalize the vector
        vector = vector / np.linalg.norm(vector)
        # multiply the vector with the distance
        vector = vector * distance
        # add the vector to the start point
        return Point_3D(
            self.start_point.x + vector[0],
            self.start_point.y + vector[1],
            self.start_point.z + vector[2],
        )

    def length(self):
        return self.start_point.distance(self.end_point)
