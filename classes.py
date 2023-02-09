from shapely.geometry import LineString

class Cable_Road:
    def __init__(self, start_point, end_point):
        self.support_height = 11
        self.min_height = 3
        self.start_point = start_point
        self.end_point = end_point
        self.line = LineString(start_point, end_point)
        self.max_deviation = 0.1
        self.q_s_self_weight_center_span = 10
        self.q_load = 80000
        self.no_collisions = False
