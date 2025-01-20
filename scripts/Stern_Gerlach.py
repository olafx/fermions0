import ctypes
from ctypes import py_object, c_double, c_size_t, c_bool, c_uint64
from pathlib import Path

# physics params
dynamics = 'zigzag'; assert dynamics in ('Bohmian', 'Pauli', 'zigzag')
sig = [100., 100., 100.] # initial Gaussian packet standard deviation
b = 1e-6 # magnetic field strength
px = 1e-1 # Gaussian packet initial x momentum
T = 1e5 # final integration time
ti, tf = T/8, 3/8*T # magnetic field initial and final time
s = '+y' # initial spinor polarization
c = { # associated spinor coefficients
  '+x': [1j*2**-.5, 1j*2**-.5], '-x': [1j*2**-.5, -1j*2**-.5],
  '+y': [2**-.5+0j, 1j*2**-.5], '-y': [1j*2**-.5, 2**-.5+0j],
  '+z': [1+0j, 0j], '-z': [0j, 1+0j]}[s]
# numerical params
n = 20 # number of samples
seed = 42 # RNG seed
# integrator params
abs_tol = 1e-12 # integrator absolute error tolerance
p_tol = 1e-3 # integrator transition probability tolerance
max_iter = int(1e7) # integrator max number of iterations
print_progress = True # integrator print progress
# plot params
plot_3D = True # simple 3D interactive plot instead of 2D plot
plot_save = False # save 2D plots instead of showing
plot_dark = True # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering

Stern_Gerlach = ctypes.CDLL(Path('out')/'Stern_Gerlach.so')
match dynamics:
  case 'Bohmian':
    Stern_Gerlach.Bohmian.restype = py_object
    Stern_Gerlach.Bohmian.argtypes = [py_object, c_double, c_double, c_double, c_double, c_double, py_object, c_size_t, c_uint64, c_double, c_size_t, c_bool]
    paths = Stern_Gerlach.Bohmian(sig, b, px, ti, tf, T, c, n, seed, abs_tol, max_iter, print_progress)
  case 'Pauli':
    Stern_Gerlach.Pauli.restype = py_object
    Stern_Gerlach.Pauli.argtypes = [py_object, c_double, c_double, c_double, c_double, c_double, py_object, c_size_t, c_uint64, c_double, c_size_t, c_bool]
    paths = Stern_Gerlach.Pauli(sig, b, px, ti, tf, T, c, n, seed, abs_tol, max_iter, print_progress)
  case 'zigzag':
    Stern_Gerlach.zigzag.restype = py_object
    Stern_Gerlach.zigzag.argtypes = [py_object, c_double, c_double, c_double, c_double, c_double, py_object, c_size_t, c_uint64, c_double, c_double, c_size_t, c_bool]
    paths = Stern_Gerlach.zigzag(sig, b, px, ti, tf, T, c, n, seed, abs_tol, p_tol, max_iter, print_progress)

import matplotlib.pyplot as plt
colors = plt.cm.tab10.colors
if plot_3D:
  import vispy.app, vispy.scene
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
  xi, xf = px*ti, px*tf

  plt.figure(f'{dynamics} xy')
  for path in paths: plt.plot(path[:,1], path[:,2], lw=.5)
  ylim = plt.gca().get_ylim()
  plt.plot([xi]*2, ylim, [xf]*2, ylim, c=fill, alpha=.5)
  plt.gca().add_patch(patches.Ellipse([0]*2, 2*sig[0], 2*sig[1], facecolor=fill, alpha=.5))
  plt.ylim(*ylim)
  plt.xlabel('$x$'); plt.ylabel('$y$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Stern_Gerlach_{dynamics}_xy.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} xz')
  for path in paths: plt.plot(path[:,1], path[:,3], lw=.5)
  zlim = plt.gca().get_ylim()
  plt.plot([xi]*2, zlim, [xf]*2, zlim, c=fill, alpha=.5)
  plt.gca().add_patch(patches.Ellipse([0]*2, 2*sig[0], 2*sig[2], facecolor=fill, alpha=.5))
  plt.ylim(*zlim)
  plt.xlabel('$x$'); plt.ylabel('$z$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Stern_Gerlach_{dynamics}_xz.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} yz')
  for path in paths: plt.plot(path[:,2], path[:,3], lw=.5)
  plt.gca().add_patch(patches.Ellipse([0]*2, 2*sig[1], 2*sig[2], facecolor=fill, alpha=.5))
  plt.xlabel('$y$'); plt.ylabel('$z$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Stern_Gerlach_{dynamics}_yz.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} timestep')
  for path in paths: plt.plot(path[1:,0]-path[:-1,0])
  plt.yscale('log')
  plt.xlim(0, max(path.shape[0] for path in paths))
  plt.grid(which='both')
  plt.xlabel('iteration'); plt.ylabel(R'$\Delta t$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Stern_Gerlach_{dynamics}_Delta_t.pdf', bbox_inches='tight')

  plt.figure(f'{dynamics} time')
  for path in paths: plt.plot(path[:,0])
  plt.yscale('log')
  plt.xlim(0, max(path.shape[0] for path in paths))
  plt.grid(which='both')
  plt.xlabel('iteration'); plt.ylabel('$t$')
  plt.tight_layout()
  if plot_save: plt.savefig(Path('out')/f'Stern_Gerlach_{dynamics}_time.pdf', bbox_inches='tight')

  if not plot_save: plt.show()
