import math
import numpy as np
from scipy.special import sph_harm, hyp1f1
from util import C_to_s

# physics params
# superposition of (n,l,m) eigenstates, can be unnormalized
state = [(1, (2, 1, 0))]
state = [(1, (2, 1, +1))]
state = [(1/4, (1, 0, 0)), (3/4, (2, 1, 0))]
# plot params
plot_3D = False # simple 3D plot instead of 2D plot
hr = 10 # half range in Bohr radii
N = 256 # resolution
plot_dark = True # plot 2D with dark background, 3D is always dark
use_tex = False # use LaTeX text rendering
sqrt_scaling = True # plot the square root of density instead
plane_init = 'xz'; assert plane_init in ('xz', 'yz', 'xy') # default plane
slice_init = 0 # default slice
plot_save = False # save 2D plot instead of showing

def psi(n, l, m, r, th, phi) -> float:
  # remove Condon-Shortley phase
  Y = sph_harm(m, l, phi, th)*(-1)**m
  f = math.factorial
  R_norm = (f(n+l)/(2*n*f(n-l-1)))**.5/f(2*l+1)*(2/n)**1.5
  rho = 2*r/n
  R = R_norm*np.exp(-rho/2)*rho**l*hyp1f1(-n+l+1,2*l+2,rho)
  return R*Y

def eval_psi_volume() -> np.ndarray:
  xyz = np.meshgrid(ls := np.linspace(-hr, hr, N), ls, ls)
  return sum(c*psi(*nlm, *C_to_s(*xyz)) for c, nlm in state)

def eval_psi_plane(plane:str, slice_) -> np.ndarray:
  a, b = np.meshgrid(ls := np.linspace(-hr, hr, N), ls)
  match plane:
    case 'xy': xyz = a, b, slice_
    case 'xz': xyz = a, slice_, b
    case 'yz': xyz = slice_, a, b
  return sum(c*psi(*nlm, *C_to_s(*xyz)) for c, nlm in state)

def run_plot_2D():
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Slider, RadioButtons
  plt.rcParams['font.size'] = 14
  plt.rcParams['figure.figsize'] = (5, 5)
  if use_tex: plt.rcParams['text.usetex'] = True
  if plot_dark: plt.style.use('dark_background')
  if plot_save:
    from pathlib import Path
    from os import makedirs
    makedirs('out', exist_ok=True)
  planes = ('xz', 'yz', 'xy')
  vmax = None
  fig, ax = plt.subplots()
  def plot(data, plane, slice_):
    nonlocal vmax
    if vmax is None: vmax = np.max(data)
    ax.clear()
    ax.set_xlabel(f'${plane[0]}$'); ax.set_ylabel(f'${plane[1]}$')
    title = f'${(set('xyz')-set(plane)).pop()}={slice_:.1f}$'
    if len(state) == 1:
      _, (n, l, m) = state[0]
      title = f'$(n,l,m)=({n},{l},{m})$\n{title}'
    ax.set_title(title)
    return ax.imshow(data, extent=(-hr, hr, -hr, hr), origin='lower', vmin=0, vmax=vmax, aspect='equal',
      cmap='cividis', interpolation='none')
  data = np.abs(eval_psi_plane(plane_init, slice_init))**(1 if sqrt_scaling else 2)
  img = plot(data, plane_init, slice_init)
  fig.colorbar(img, ax=ax, fraction=.046, pad=.04).set_label(R'$\sqrt{\rho}$' if sqrt_scaling else R'$\rho$')
  ax.set_box_aspect(1)
  plt.tight_layout()
  if not plot_save:
    slice_slider_ax, plane_radio_ax = plt.axes([.3, 0, .55, .03]), plt.axes([0, 0, .15, .15])
    slice_slider = Slider(slice_slider_ax, '', -hr, hr, valinit=slice_init, dragging=not use_tex, valfmt='%.2f')
    slice_slider.valtext.set_visible(False)
    plane_radio = RadioButtons(plane_radio_ax, planes)
    def update(_):
      plane, slice_ = plane_radio.value_selected, slice_slider.val
      data = np.abs(eval_psi_plane(plane, slice_))**(1 if sqrt_scaling else 2)
      plot(data, plane, slice_)
      fig.canvas.draw_idle()
    slice_slider.on_changed(update)
    plane_radio.on_clicked(update)
  if plot_save:
    plt.savefig(Path('out')/f'H_Sch_sol.pdf', bbox_inches='tight')
  else: plt.show()

def run_plot_3D():
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas('', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
  data = np.abs(eval_psi_volume()).astype(np.float32)**(1 if sqrt_scaling else 2)
  volume = vispy.scene.visuals.Volume(data, parent=view.scene)
  volume.cmap = 'magma'
  volume.method = 'mip'
  view.camera.set_range()
  canvas.app.run()

if __name__ == '__main__':
  state = [(c/sum(abs(c)**2 for c, _ in state)**.5, nlm) for c, nlm in state]
  run_plot_3D() if plot_3D else run_plot_2D()
