import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path
import numpy as np
from util import validate_circular_paths, filter_r_max

# physics params
orbital = '210'; assert orbital in ('100', '200', '210', '211', '2p_x', '100_210')
a = 1 # Bohr radius
om1 = 1/(2*a**2) # frequency associated to n=1 energy
om12 = 3/4*om1 # frequency associated to n=1 to n=2 transition energy
tf = 3e2/om1 # final time
c_100_210 = [(1/4)**.5+0j, (3/4)**.5+0j] # coefficients for the 100-210 hybrid orbital
# numerical params
n_samples = int(1e4) # number of samples
n_paths = 5 # number of samples for which to also plot trajectories
anal_path_n = int(1e5) # number of vertices for analytically evaluated paths
stat_curve_n = int(1e4) # stationary curve number of vertices
stat_curve_len = 15 # stationary curve length in Bohr radii
seed = 42 # RNG seed
# MC params
xi_max = 9. # maximum distance to sample in units of Bohr radius
print_MC_eff = True # print Monte Carlo sampling efficiency
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance (but we undimensionalize)
max_iter = int(1e6) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_3D = False # simple 3D interactive plot instead of 2D plot
plot_save = True # save 2D plots instead of showing
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
lw = 2 # linewidth
ms = .5 # markersize
plot_ticks = False # plot coordinate ticks

H_Sch = ctypes.CDLL(Path('out')/'H_Sch.so')
match orbital:
  case '100':
    H_Sch.sample_100.restype = py_object
    H_Sch.sample_100.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_100.restype = py_object
    H_Sch.Pauli_100.argtypes = [c_double, c_double, py_object, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_100(a, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_100(a, om1, samples[:n_paths,:], tf, anal_path_n, print_progress)
    validate_circular_paths(paths, samples, lambda r: 2*om1/(r/a))
  case '200':
    H_Sch.sample_200.restype = py_object
    H_Sch.sample_200.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_200.restype = py_object
    H_Sch.Pauli_200.argtypes = [c_double, c_double, py_object, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_200(a, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_200(a, om1, samples[:n_paths,:], tf, anal_path_n, print_progress)
    validate_circular_paths(paths, samples, lambda r: -om1/(xi := r/a)*(1/(xi/2-1)-1))
  case '210':
    H_Sch.sample_210.restype = py_object
    H_Sch.sample_210.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_210.restype = py_object
    H_Sch.Pauli_210.argtypes = [c_double, c_double, py_object, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_210(a, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_210(a, om1, samples[:n_paths,:], tf, anal_path_n, print_progress)
    validate_circular_paths(paths, samples, lambda r: om1/(r/a))
  case '211':
    H_Sch.sample_211.restype = py_object
    H_Sch.sample_211.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_211.restype = py_object
    H_Sch.Pauli_211.argtypes = [c_double, c_double, py_object, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_211(a, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_211(a, om1, samples[:n_paths,:], tf, anal_path_n, print_progress)
    validate_circular_paths(paths, samples, lambda r: om1/(r/a))
  case '2p_x':
    H_Sch.sample_2p_x.restype = py_object
    H_Sch.sample_2p_x.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_2p_x.restype = py_object
    H_Sch.Pauli_2p_x.argtypes = [c_double, c_double, py_object, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_2p_x(a, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_2p_x(a, om1, samples[:n_paths,:], tf, abs_tol, max_iter, print_progress)
    # stationary curve, x>0 z>0 quadrant only
    stat = np.array([
      x := np.linspace(2*a, stat_curve_len*a, stat_curve_n//4),
      np.zeros_like(x),
      x*((.5*x/a)**2-1)**.5])
    assert stat[2,-1] >= stat_curve_len*a/2 # should cover the domain
  case '100_210':
    H_Sch.sample_100_210.restype = py_object
    H_Sch.sample_100_210.argtypes = [c_double, py_object, c_size_t, c_uint64, c_bool]
    H_Sch.Pauli_100_210.restype = py_object
    H_Sch.Pauli_100_210.argtypes = [c_double, c_double, py_object, py_object, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_100_210(a, c_100_210, n_samples, seed, False)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Sch.Pauli_100_210(a, om12, c_100_210, samples[:n_paths,:], tf, abs_tol, max_iter, print_progress)
if print_MC_eff: print(f'sampling eff {n_samples/attempts:.2e}')

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_3D:
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas(f'{orbital} Pauli', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'fly'
  for i, path in enumerate(paths):
    vispy.scene.Line(path[:,1:], color=colors[i % len(colors)], parent=view.scene)
  if orbital == '2p_x':
    x, y, z = stat
    for _ in ((x, y, z), (x, y, -z), (-x, y, z), (-x, y, -z)):
      vispy.scene.Line(np.array(_).T, color=(1, 0, 0, .5), parent=view.scene)
  markers = vispy.scene.Markers()
  markers.set_data(samples, size=2)
  view.add(markers)
  view.camera.set_range()
  canvas.app.run()
else:
  if plot_save:
    from os import makedirs
    makedirs('out', exist_ok=True)
  plt.rcParams['font.size'] = 18
  plt.rcParams['figure.figsize'] = (5, 5)
  if use_tex: plt.rcParams['text.usetex'] = True
  if plot_dark:
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
  fill = 'white' if plot_dark else 'black'
  make_square = lambda lim1, lim2: (-(_ := abs(max(lim1+lim2, key=abs))), _)

  plt.figure(f'{orbital} Pauli xy')
  plt.plot(samples[:,0], samples[:,1], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,2], lw=lw)
  if orbital == '2p_x':
    x, y, z = stat
    plt.plot(x, y, '--', -x, y, '--', c='red', alpha=.5)
  if xi_max is None:
    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(xlim, ylim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  if not plot_ticks:
    plt.xticks([], [])
    plt.yticks([], [])
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} Pauli xz')
  plt.plot(samples[:,0], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,3], lw=lw)
  if orbital == '2p_x': plt.plot(x, z, '--', x, -z, '--', -x, z, '--', -x, -z, '--', c='red', alpha=.5)
  xlim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
  if xi_max is None:
    plt.xlim(*(_ := make_square(xlim, zlim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  if not plot_ticks:
    plt.xticks([], [])
    plt.yticks([], [])
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} Pauli yz')
  plt.plot(samples[:,1], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,2], path[:,3], lw=lw)
  if orbital == '2p_x': plt.plot(0, 0, 'o', c='red', alpha=.5, ms=2)
  if xi_max is None:
    ylim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(ylim, zlim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$y$'); plt.ylabel('$z$')
  if not plot_ticks:
    plt.xticks([], [])
    plt.yticks([], [])
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_yz.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
