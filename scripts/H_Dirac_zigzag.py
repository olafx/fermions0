import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path

orbital = '2_1_3o2_p1o2'; assert orbital in ('1_0_1o2_p1o2', '2_1_3o2_p1o2', '2_1_3o2_p3o2')
a = 1 # Bohr radius
om1 = 1/(2*a**2) # frequency associated to nonrelativistic n=1 energy
# M factor at least 1
# M = 1/(a*om1) # dimensionless mass, or speed of light
# alpha = 2/M
alpha = .5
M = 2/alpha
tf = 2e2/om1 # final time
# numerical params
n_samples = int(1e4) # number of samples
n_paths = 2 # number of samples for which to also plot trajectories
seed = 42 # RNG seed
# MC params
xi_max = 10. # maximum distance to sample in units of Bohr radius
print_MC_eff = True # print Monte Carlo sampling efficiency
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance (but we undimensionalize)
p_tol = 1e-4 # integrator transition probability tolerance
max_iter = int(1e7) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_3D = False # simple 3D interactive plot instead of 2D plot
plot_save = False # save 2D plots instead of showing
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
lw = .5 # linewidth
ms = .5 # markersize

H_Dirac = ctypes.CDLL(Path('out')/'H_Dirac.so')
match orbital:
  case '1_0_1o2_p1o2':
    H_Dirac.sample_1_0_1o2_p1o2.restype = py_object
    H_Dirac.sample_1_0_1o2_p1o2.argtypes = [c_double, c_double, c_size_t, c_uint64, c_bool]
    H_Dirac.zigzag_1_0_1o2_p1o2.restype = py_object
    H_Dirac.zigzag_1_0_1o2_p1o2.argtypes = [c_double, c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Dirac.sample_1_0_1o2_p1o2(a, alpha, n_samples, seed, True)
    paths = H_Dirac.zigzag_1_0_1o2_p1o2(a, alpha, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_1_3o2_p1o2':
    H_Dirac.sample_2_1_3o2_p1o2.restype = py_object
    H_Dirac.sample_2_1_3o2_p1o2.argtypes = [c_double, c_double, c_size_t, c_uint64, c_bool]
    H_Dirac.zigzag_2_1_3o2_p1o2.restype = py_object
    H_Dirac.zigzag_2_1_3o2_p1o2.argtypes = [c_double, c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Dirac.sample_2_1_3o2_p1o2(a, alpha, n_samples, seed, True)
    paths = H_Dirac.zigzag_2_1_3o2_p1o2(a, alpha, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
  case '2_1_3o2_p3o2':
    H_Dirac.sample_2_1_3o2_p3o2.restype = py_object
    H_Dirac.sample_2_1_3o2_p3o2.argtypes = [c_double, c_double, c_size_t, c_uint64, c_bool]
    H_Dirac.zigzag_2_1_3o2_p3o2.restype = py_object
    H_Dirac.zigzag_2_1_3o2_p3o2.argtypes = [c_double, c_double, c_double, c_double, py_object, c_double, c_uint64, c_double, c_double, c_size_t, c_bool]
    attempts, samples = H_Dirac.sample_2_1_3o2_p3o2(a, alpha, n_samples, seed, True)
    paths = H_Dirac.zigzag_2_1_3o2_p3o2(a, alpha, om1, M, samples[:n_paths,:], tf, seed, abs_tol, p_tol, max_iter, print_progress)
if print_MC_eff: print(f'sampling eff {n_samples/attempts:.2e}')

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_3D:
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
  plt.plot(samples[:,0], samples[:,1], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,2], lw=lw)
  if xi_max is None:
    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(xlim, ylim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Dirac_H_zigzag_{orbital}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples xz')
  plt.plot(samples[:,0], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,1], path[:,3], lw=lw)
  if xi_max is None:
    xlim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(xlim, zlim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Dirac_H_zigzag_{orbital}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} samples yz')
  plt.plot(samples[:,1], samples[:,2], 'o', c=fill, ms=ms)
  for path in paths: plt.plot(path[:,2], path[:,3], lw=lw)
  if xi_max is None:
    ylim, zlim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*(_ := make_square(ylim, zlim))); plt.ylim(*_)
  else:
    plt.xlim(-xi_max, xi_max); plt.ylim(-xi_max, xi_max)
  plt.xlabel('$y$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Dirac_H_zigzag_{orbital}_yz.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
