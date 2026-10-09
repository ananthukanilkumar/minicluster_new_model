import os
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
X = np.load("X.npy")          # shape (1000, 3)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, alpha=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
#plt.show()

import plotly.graph_objects as go
go.Figure(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],
                       mode='markers', marker=dict(size=2))).write_html("p.html")