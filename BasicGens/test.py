import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shapely.geometry import Point, LineString

times = np.array([1.0, 2.0, 3.0])
data = np.array([[0.0, 0.5], [0.5, 0.1], [1.0, 1.2]])

# pos = nept.Position(data, times)

line = LineString([(0.0, 0.0), (1.0, 1.0)])
ideal_path = line
zpos = []
for point_x, point_y in zip(data[:, 0], data[:, 1]):
    zpos.append(ideal_path.project(Point(point_x, point_y)))
zpos = np.array(zpos)


# linear = pos.linearize(line)

assert np.allclose(linear.x, np.array([0.35355339, 0.42426407, 1.41421356]))
assert np.allclose(linear.time, np.array([1.0, 2.0, 3.0]))
