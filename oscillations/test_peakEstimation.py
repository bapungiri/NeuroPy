import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg


t = np.linspace(0, 10, 10000)
y = np.sin(2 * np.pi * 4 * t)

grad = np.gradient(y)
zero_crossings = np.where(np.diff(np.sign(grad)))[0]
cross_sign = np.zeros(len(zero_crossings))

for i, ind in enumerate(zero_crossings):
    if grad[ind - 1] < grad[ind + 1]:
        cross_sign[i] = 1

up = zero_crossings[np.where(cross_sign == 1)[0]]
down = zero_crossings[np.where(cross_sign == 0)[0]]

plt.clf()
plt.plot(t, y)
plt.plot(t, grad)
plt.plot(t[zero_crossings], grad[zero_crossings], "r.")
plt.plot(t[up], y[up], "g.")
plt.plot(t[down], y[down], "k.")

a = 3
b = -0.1

if (a > 2 and b < 0) or (a > 1 and b < -1.5):
    print(a, b)
