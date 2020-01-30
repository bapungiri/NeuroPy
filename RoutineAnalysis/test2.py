import time

import numpy as np
import matplotlib.pyplot as plt


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


# Copy to clipboard

# Define a triangle by clicking three points

plt.clf()
plt.setp(plt.gca(), autoscale_on=True)

plt.plot(np.arange(1, 21), np.random.rand(20, 1))

tellme("You will define a triangle, click to begin")

plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 2:
        tellme("Select 2 edges with mouse")
        pts = np.asarray(plt.ginput(2, timeout=-1))
        if len(pts) < 2:
            tellme("Too few points, starting over")
            time.sleep(1)  # Wait a second

        pts = np.asarray(
            [[pts[0, 0], 1], [pts[0, 0], 0], [pts[1, 0], 0], [pts[1, 0], 1]]
        )

    ph = plt.fill(pts[:, 0], pts[:, 1], "r", lw=2)

    tellme("Happy? Key click for yes, mouse click for no")

    if plt.waitforbuttonpress():
        break

    # Get rid of fill
    for p in ph:
        p.remove()

# Copy to clipboard

# Now contour according to distance from triangle corners - just an example

# Define a nice function of distance from individual pts
def f(x, y, pts):
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1 / (np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2))
    return 1 / z


X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
Z = f(X, Y, pts)

CS = plt.contour(X, Y, Z, 20)

tellme("Use mouse to select contour label locations, middle button to finish")
CL = plt.clabel(CS, manual=True)

# Copy to clipboard

# Now do a zoom

tellme("Now do a nested zoom, click to begin")
plt.waitforbuttonpress()

while True:
    tellme("Select two corners of zoom, middle mouse button to finish")
    pts = plt.ginput(2, timeout=-1)
    if len(pts) < 2:
        break
    (x0, y0), (x1, y1) = pts
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

tellme("All Done!")
plt.show()
