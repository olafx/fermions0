import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path
from util import filter_r_max

# physics params
orbital = '2_1_1o2_p1o2'; assert orbital in ('1_0_1o2_p1o2', '2_0_1o2_p1o2', '2_1_1o2_p1o2', '2_1_3o2_p1o2', '2_1_3o2_p3o2')
a = 1 # Bohr radius
om1 = 1/(2*a**2) # frequency associated to n=1 energy
alpha = .5 # fine structure constant
M = 2/alpha # dimensionless mass, or speed of light
tf = 3e2/om1 # final time
# numerical params
n_samples = int(1e4) # number of samples
n_paths = 2 # number of samples for which to also plot trajectories
seed = 42 # RNG seed
# MC params
xi_max = 9. # maximum distance to sample in units of Bohr radius
print_MC_eff = True # print Monte Carlo sampling efficiency
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance (but we undimensionalize)
p_tol = 1e-3 # integrator transition probability tolerance
max_iter = int(1e7) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_3D = False # simple 3D interactive plot instead of 2D plot
plot_save = True # save 2D plots instead of showing
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
lw = 1 # linewidth
ms = .5 # markersize
plot_ticks = False # plot coordinate ticks

H_Pauli = ctypes.CDLL(Path('out')/'H_Pauli.so')
match orbital:
  case '1_0_1o2_p1o2':
    H_Pauli.sample_1_0_1o2_p1o2.restype = py_object
    H_Pauli.sample_1_0_1o2_p1o2.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Pauli.zigzag_1_0_1o2_p1o2.restype = py_object
    H_Pauli.zigzag_1_0_1o2_p1o2.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Pauli.sample_1_0_1o2_p1o2(a, n_samples, seed, True)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Pauli.zigzag_1_0_1o2_p1o2(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_0_1o2_p1o2':
    H_Pauli.sample_2_0_1o2_p1o2.restype = py_object
    H_Pauli.sample_2_0_1o2_p1o2.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Pauli.zigzag_2_0_1o2_p1o2.restype = py_object
    H_Pauli.zigzag_2_0_1o2_p1o2.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Pauli.sample_2_0_1o2_p1o2(a, n_samples, seed, True)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Pauli.zigzag_2_0_1o2_p1o2(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_1_1o2_p1o2':
    H_Pauli.sample_2_1_1o2_p1o2.restype = py_object
    H_Pauli.sample_2_1_1o2_p1o2.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Pauli.zigzag_2_1_1o2_p1o2.restype = py_object
    H_Pauli.zigzag_2_1_1o2_p1o2.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Pauli.sample_2_1_1o2_p1o2(a, n_samples, seed, True)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Pauli.zigzag_2_1_1o2_p1o2(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_1_3o2_p1o2':
    H_Pauli.sample_2_1_3o2_p1o2.restype = py_object
    H_Pauli.sample_2_1_3o2_p1o2.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Pauli.zigzag_2_1_3o2_p1o2.restype = py_object
    H_Pauli.zigzag_2_1_3o2_p1o2.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Pauli.sample_2_1_3o2_p1o2(a, n_samples, seed, True)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Pauli.zigzag_2_1_3o2_p1o2(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_1_3o2_p3o2':
    H_Pauli.sample_2_1_3o2_p3o2.restype = py_object
    H_Pauli.sample_2_1_3o2_p3o2.argtypes = [c_double, c_size_t, c_uint64, c_bool]
    H_Pauli.zigzag_2_1_3o2_p3o2.restype = py_object
    H_Pauli.zigzag_2_1_3o2_p3o2.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Pauli.sample_2_1_3o2_p3o2(a, n_samples, seed, True)
    samples = filter_r_max(samples, xi_max*a)
    paths = H_Pauli.zigzag_2_1_3o2_p3o2(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
if print_MC_eff: print(f'sampling eff {n_samples/attempts:.2e}')

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_3D:
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas(f'{orbital} zigzag', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'fly'
  for i, path in enumerate(paths):
    vispy.scene.Line(path[:,1:4], color=colors[i % len(colors)], parent=view.scene)
  markers = vispy.scene.Markers()
  markers.set_data(samples[:,:3], size=2)
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

  plt.figure(f'{orbital} samples xy')
  plt.plot(samples[:,0], samples[:,1], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,2], lw=lw)
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
  if plot_save: plt.savefig(Path('out')/f'Pauli_H_zigzag_{orbital}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples xz')
  plt.plot(samples[:,0], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,3], lw=lw)
  if xi_max is None:
    xlim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(xlim, zlim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  if not plot_ticks:
    plt.xticks([], [])
    plt.yticks([], [])
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Pauli_H_zigzag_{orbital}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples yz')
  plt.plot(samples[:,1], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,2], path[:,3], lw=lw)
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
  if plot_save: plt.savefig(Path('out')/f'Pauli_H_zigzag_{orbital}_yz.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
