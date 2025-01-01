import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool
import numpy as np
from util import equal_span_partition, dphidtau, s_to_C, C_to_s
from pathlib import Path

# physics params
orbital = '100_210_sigmoid'; assert orbital in ('100_210_sigmoid', '100_210_Rabi')
a = 5.29e-11 # Bohr radius
om0 = 1.55e16 # frequency associated to n=1 to n=2 transition energy
Om = 1.55e12 # frequency associated to Rabi perturbation frequency
nu = -5.1e12 # frequency associated to Rabi perturbation strength
tf = np.pi/(Om**2+nu**2)**.5 # final time
tf = 10000/om0
r0, th0, phi0 = 4*a, 1, 0 # initial condition
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance (but we undimensionalize)
max_iter = int(1e6) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_interactive_3D = False # simple 3D interactive plot instead of 2D plot
plot_save = False # save 2D plots instead of showing
plot_dark = True # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
n_colors = 100 # number of different colors for time-dependence

H_Sch = ctypes.CDLL(Path('out')/'H_Sch.so')
match orbital:
  case '100_210_sigmoid':
    sig = (Om**2+nu**2)**.5
    H_Sch.Pauli_100_210_sigmoid.restype = py_object
    H_Sch.Pauli_100_210_sigmoid.argtypes = [c_double, c_double, c_double, py_object, c_double, c_double, c_size_t, c_bool]
    path = H_Sch.Pauli_100_210_sigmoid(a, om0, sig, np.array([[s_to_C(r0, th0, phi0)]]), tf, abs_tol, max_iter, print_progress)[0]
  case '100_210_Rabi':
    H_Sch.Pauli_100_210_Rabi.restype = py_object
    H_Sch.Pauli_100_210_Rabi.argtypes = [c_double, c_double, c_double, c_double, py_object, c_double, c_double, c_size_t, c_bool]
    path = H_Sch.Pauli_100_210_Rabi(a, om0, Om, nu, np.array([[s_to_C(r0, th0, phi0)]]), tf, abs_tol, max_iter, print_progress)[0]
part_ids = equal_span_partition(path[:,0], n_colors)
filter_under=1
filter, dphidtau_ = dphidtau(path, 1/om0, filter_under=filter_under)
path_filtered = path[filter]
r, th, phi = C_to_s(*path_filtered[:,1:].T)
tau = path_filtered[:,0]*om0
dphidtau_target = (8/3/(xi := r/a), 4/3/xi*(1-1/xi))

from matplotlib import colormaps
from matplotlib.colors import Normalize
cmap = 'plasma' if plot_dark else 'viridis'
color = lambda i: colormaps[cmap](Normalize(0, n_colors)(i))
if plot_interactive_3D:
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas(f'{orbital} Pauli', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
  for part_i, (left, right) in enumerate(part_ids):
    vispy.scene.Line(path[left:right,1:], color=color(part_i), parent=view.scene)
  view.camera.set_range()
  canvas.app.run()
else:
  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  if plot_save:
    from os import makedirs
    makedirs('out', exist_ok=True)
  plt.rcParams['font.size'] = 14
  plt.rcParams['figure.figsize'] = (5, 5)
  if use_tex: plt.rcParams['text.usetex'] = True
  if plot_dark: plt.style.use('dark_background')
  lim = (-(_ := max(np.max(np.abs(path[:,i])) for i in range(3))*1.1), _)
  sm = plt.cm.ScalarMappable(Normalize(path[0,0], path[-1,0]), cmap)
  def gen_lc(i, j):
    lc, colors = [], []
    for part_i, (left, right) in enumerate(part_ids):
      lc += [np.column_stack((path[left:right,i], path[left:right,j]))]
      colors += [color(part_i)]
    return LineCollection(lc, colors=colors, linewidths=.5)
  fill = 'white' if plot_dark else 'black'

  plt.figure(f'{orbital} Pauli xy')
  plt.gca().add_collection(gen_lc(1, 2))
  plt.colorbar(sm, ax=plt.gca(), fraction=.046, pad=.04).set_label('$t$')
  plt.xlim(*lim); plt.ylim(*lim)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} Pauli xz')
  plt.gca().add_collection(gen_lc(1, 3))
  plt.colorbar(sm, ax=plt.gca(), fraction=.046, pad=.04).set_label('$t$')
  plt.xlim(*lim); plt.ylim(*lim)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{orbital} Pauli yz')
  plt.gca().add_collection(gen_lc(2, 3))
  plt.colorbar(sm, ax=plt.gca(), fraction=.046, pad=.04).set_label('$t$')
  plt.xlim(*lim); plt.ylim(*lim)
  plt.xlabel('$y$'); plt.ylabel('$z$')
  plt.gca().set_box_aspect(1)
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_yz.pdf', bbox_inches='tight')

  plt.rcParams['figure.figsize'] = (6.4, 4.8)
  plt.figure(f'{orbital} Pauli orbital rate')
  plt.plot(tau, dphidtau_[filter], c=fill)
  plt.plot(tau, dphidtau_target[0], tau, dphidtau_target[1], c='red',alpha=.5)
  plt.xlim(tau[0], tau[-1]); plt.ylim(-filter_under, filter_under)
  plt.xlabel(R'$\tau$'); plt.ylabel(R'$d\phi/d\tau$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Sch_H_Pauli_{orbital}_rate.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
