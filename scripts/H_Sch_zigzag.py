import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path

# physics params
orbital = '100_210'; assert orbital in ('100', '200', '210', '211', '2p_x', '100_210')
a = 1 # Bohr radius
om1 = 1/(2*a**2) # frequency associated to n=1 energy
om12 = 3/4*om1 # frequency associated to n=1 to n=2 transition energy
M = 1/(a*om1) # dimensionless mass, or speed of light
tf = 2e2/om1 # final time
c_100_210 = [2**-.5, 2**-.5] # real coefficients for the 100-210 hybrid orbital
c_100_210 = [(1/4)**.5, (3/4)**.5]
sig_100_210 = om1/1e2 # transition rate for the 100-210-sigmoid hybrid orbital
# numerical params
n_samples = int(1e4) # number of samples
n_paths = 1 # number of samples for which to also plot trajectories
seed = 47 # RNG seed
# MC params
xi_max = 20. # maximum distance to sample in units of Bohr radius
print_MC_eff = True # print Monte Carlo sampling efficiency
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance (but we undimensionalize)
p_tol = 1e-3 # integrator transition probability tolerance
max_iter = int(1e7) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_interactive_3D = True # simple 3D interactive plot instead of 2D plot
plot_save = False # save 2D plots instead of showing
plot_dark = True # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering

H_Sch = ctypes.CDLL(Path('out')/'H_Sch.so')
match orbital:
  case '100':
    H_Sch.sample_100.restype = py_object
    H_Sch.sample_100.argtypes = [c_double, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_100.restype = py_object
    H_Sch.zigzag_100.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_100(a, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_100(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '200':
    H_Sch.sample_200.restype = py_object
    H_Sch.sample_200.argtypes = [c_double, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_200.restype = py_object
    H_Sch.zigzag_200.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_200(a, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_200(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '210':
    H_Sch.sample_210.restype = py_object
    H_Sch.sample_210.argtypes = [c_double, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_210.restype = py_object
    H_Sch.zigzag_210.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_210(a, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_210(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '211':
    H_Sch.sample_211.restype = py_object
    H_Sch.sample_211.argtypes = [c_double, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_211.restype = py_object
    H_Sch.zigzag_211.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_211(a, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_211(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2p_x':
    H_Sch.sample_2p_x.restype = py_object
    H_Sch.sample_2p_x.argtypes = [c_double, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_2p_x.restype = py_object
    H_Sch.zigzag_2p_x.argtypes = [c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_2p_x(a, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_2p_x(a, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '100_210':
    H_Sch.sample_100_210.restype = py_object
    H_Sch.sample_100_210.argtypes = [c_double, py_object, c_size_t, c_uint64, c_double, c_bool]
    H_Sch.zigzag_100_210.restype = py_object
    H_Sch.zigzag_100_210.argtypes = [c_double, c_double, c_double, py_object, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Sch.sample_100_210(a, c_100_210, n_samples, seed, xi_max, True)
    paths = H_Sch.zigzag_100_210(a, om12, M, c_100_210, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
if print_MC_eff: print(f'sampling eff {n_samples/attempts:.2e}')

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_interactive_3D:
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas(f'{orbital} zigzag', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
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
  plt.rcParams['font.size'] = 14
  plt.rcParams['figure.figsize'] = (5, 5)
  if use_tex: plt.rcParams['text.usetex'] = True
  if plot_dark:
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
  fill = 'white' if plot_dark else 'black'
  make_square = lambda lim1, lim2: (-(_ := abs(max(lim1+lim2, key=abs))), _)

  plt.figure(f'{orbital} samples xy')
  plt.plot(samples[:,0], samples[:,1], 'o', c=fill, ms=.5)
  for path in paths: plt.plot(path[:,1], path[:,2], lw=.5)
  xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
  plt.xlim(*(_ := make_square(xlim, ylim))); plt.ylim(*_)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_zigzag_{orbital}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples xz')
  plt.plot(samples[:,0], samples[:,2], 'o', c=fill, ms=.5)
  for path in paths: plt.plot(path[:,1], path[:,3], lw=.5)
  xlim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
  plt.xlim(*(_ := make_square(xlim, zlim))); plt.ylim(*_)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_zigzag_{orbital}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples yz')
  plt.plot(samples[:,1], samples[:,2], 'o', c=fill, ms=.5)
  for path in paths: plt.plot(path[:,2], path[:,3], lw=.5)
  ylim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
  plt.xlim(*(_ := make_square(ylim, zlim))); plt.ylim(*_)
  plt.xlabel('$y$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_zigzag_{orbital}_yz.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
