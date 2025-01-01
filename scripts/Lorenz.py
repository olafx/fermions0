import ctypes
from ctypes import py_object, c_double
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import vispy.app, vispy.scene
from pathlib import Path

n = 6 # number of initial conditions
rho, sigma, beta = 28, 18, 8/3 # Lorenz attractor params
delta = .2 # distance between initial conditions
tf = 40 # final time
abs_tol = 1e-12 # integrator absolute error tolerance
plot_dark = True # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering

L = ctypes.CDLL(Path('out')/'Lorenz.so')
L.Lorenz.restype = py_object
L.Lorenz.argtypes = [c_double, c_double, c_double, py_object, c_double, c_double]
paths_0 = np.zeros((n, 3), dtype=np.float64)
paths_0[:,0] = np.linspace(1+delta/2, 1-delta/2, n)
paths = L.Lorenz(rho, sigma, beta, paths_0, tf, abs_tol)

canvas = vispy.scene.SceneCanvas('Lorenz', keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, path in enumerate(paths): vispy.scene.Line(path[:,1:], color=colors[i % len(colors)], parent=view.scene)
view.camera.set_range()
canvas.app.run()

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (6.4, 4.8)
if use_tex: plt.rcParams['text.usetex'] = True
if plot_dark: plt.style.use('dark_background')
color = lambda i: colormaps['Reds_r' if plot_dark else 'inferno_r'](Normalize(0, len(paths)-1)(i))
plt.figure('timestep')
for i, path in enumerate(paths): plt.plot(path[1:,0]-path[:-1,0], c=color(i))
plt.xlabel('iteration'); plt.ylabel(R'$\Delta t$')
plt.yscale('log')
plt.xlim(0, max(path.shape[0] for path in paths))
plt.grid(which='both')
plt.tight_layout()
plt.show()
