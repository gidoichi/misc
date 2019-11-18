#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# データの取得
z = []
x = []
y = []
while 1:
    try:
        line = input().split()
        z.append(float(line[0]))
        x.append(float(line[1]))
        y.append(float(line[2]))
    except EOFError:
        break

# z座標に応じた色付け
# https://teratail.com/questions/126209
#   or
# https://stackoverflow.com/questions/11950375/apply-color-map-to-mpl-toolkits-mplot3d-axes3d-bar3d
offset = z + np.abs(min(z))
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
clrs = cm.cool(norm(fracs))

# グラフの描画
ax = Axes3D(plt.figure())
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.scatter(x, y, z, color=clrs, s=10)
plt.savefig("report1_plot.pdf")
