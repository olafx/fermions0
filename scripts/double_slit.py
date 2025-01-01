import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path

# physics params
dynamics = 'zigzag'; assert dynamics in ('Bohmian', 'Pauli', 'zigzag')
m = .3 # mass
sig = [50., sig_y := 20., 5*sig_y] # initial Gaussian packet standard deviations
d = 6*sig_y # distance between slits
dist = 1e3 # distance from slits to detector (where integration stops)
vx = .1 # Gaussian packet initial x velocity
s = [3**-.5, 3**-.5, 3**-.5] # spin polarization
scale_system_with_mass = True # scale the system with mass, approx fixed TOF
# numerical params
n = 40 # number of samples
seed = 42 # RNG seed
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance
p_tol = 1e-3 # integrator transition probability tolerance
max_iter = int(1e6) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_interactive_3D = False # simple 3D interactive plot instead of 2D plot
plot_save = False # save 2D plots instead of showing
plot_dark = True # plot 2D with dark background, 3D is always dark
zoom_slits_yz = True # zoom in on the slits in the yz plot
use_tex = True # use LaTeX text rendering

if scale_system_with_mass:
  vx /= m
  dist /= m
  for sig_ in sig: sig_ /= m

double_slit = ctypes.CDLL(Path('out')/'double_slit.so')
match dynamics:
  case 'Bohmian':
    double_slit.Bohmian.restype = py_object
    double_slit.Bohmian.argtypes = [py_object, c_double, c_double, c_double, c_double, c_size_t, c_uint64, c_double, c_size_t, c_bool]
    paths = double_slit.Bohmian(sig, d, vx, dist, m, n, seed, abs_tol, max_iter, print_progress)
  case 'Pauli':
    double_slit.Pauli.restype = py_object
    double_slit.Pauli.argtypes = [py_object, c_double, py_object, c_double, c_double, c_double, c_size_t, c_uint64, c_double, c_size_t, c_bool]
    paths = double_slit.Pauli(sig, d, s, vx, dist, m, n, seed, abs_tol, max_iter, print_progress)
  case 'zigzag':
    double_slit.zigzag.restype = py_object
    double_slit.zigzag.argtypes = [py_object, c_double, py_object, c_double, c_double, c_double, c_size_t, c_uint64, c_double, c_double, c_size_t, c_bool]
    paths = double_slit.zigzag(sig, d, s, vx, dist, m, n, seed, abs_tol, p_tol, max_iter, print_progress)

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_interactive_3D:
  import vispy.app
  import vispy.scene
  canvas = vispy.scene.SceneCanvas(dynamics, keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
  for i, path in enumerate(paths): vispy.scene.Line(path[:,1:4], color=colors[i % len(colors)], parent=view.scene)
  view.camera.set_range()
  canvas.app.run()
else:
  import matplotlib.patches as patches
  plt.rcParams['font.size'] = 14
  plt.rcParams['figure.figsize'] = (6.4, 4.8)
  if use_tex: plt.rcParams['text.usetex'] = True
  if plot_dark:
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
  if plot_save:
    from os import makedirs
    makedirs('out', exist_ok=True)
  fill = 'white' if plot_dark else 'black'

  plt.figure(f'{dynamics} xy')
  for path in paths: plt.plot(path[:,1], path[:,2], lw=.5)
  plt.plot([0]*2, (+d-sig[1], +d+sig[1]), c=fill, alpha=.5)
  plt.plot([0]*2, (-d-sig[1], -d+sig[1]), c=fill, alpha=.5)
  ylim = plt.gca().get_ylim()
  plt.plot([dist]*2, ylim, c=fill, alpha=.5)
  plt.ylim(*ylim)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'double_slit_{dynamics}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} xz')
  for path in paths: plt.plot(path[:,1], path[:,3], lw=.5)
  plt.plot([0]*2, [-sig[2], sig[2]], c=fill, alpha=.5)
  zlim = plt.gca().get_ylim()
  plt.plot([dist]*2, zlim, c=fill, alpha=.5)
  plt.ylim(*zlim)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'double_slit_{dynamics}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} yz')
  for path in paths: plt.plot(path[:,2], path[:,3], lw=.5)
  for _ in (-d, +d):
    plt.gca().add_patch(patches.Rectangle([_-sig[1], -sig[2]], 2*sig[1], 2*sig[2], facecolor=fill, alpha=.5))
  plt.xlabel('$y$'); plt.ylabel('$z$')
  if zoom_slits_yz:
    plt.xlim(-2*d, 2*d)
    plt.ylim(-2*sig[2], 2*sig[2])
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'double_slit_{dynamics}_yz.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
