import math
import numpy as np
from scipy.special import sph_harm, hyp1f1
from util import C_to_s

# physics params
# superposition of (n,l,j,m) eigenstates, can be unnormalized
# state = [(1, (2, 1, 1/2, +1/2)), (1j, (3, 0, 1/2, -1/2)), (-2, (4, 3, 5/2, -1/2))]
state = [(1, (1, 0, 1/2, +1/2))]
alpha = 0.3 # fine structure constant
# plot params
plot_3D = False # simple 3D plot instead of 2D plot
quiver_2D = True
hr = 10 # half range in Bohr radii
N = 512 # resolution
N_quiver = 16 # quiver resolution
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
sqrt_scaling = True # plot the square root of density instead
plane_init = 'xz'; assert plane_init in ('xz', 'yz', 'xy') # default plane
slice_init = 0 # default slice
plot_save = False # save 2D plot instead of showing
plot_title = False # plot the state in the title

def psi(n, l, j, m, alpha, xi, th, phi) -> tuple[float,float,float,float]:
  p = j-l == 1/2
  pm = 2*p-1
  a = ((l+pm*m+1/2)/(2*l+1))**.5
  b = -pm*((l-pm*m+1/2)/(2*l+1))**.5
  c = -((l-pm*(m-1)+1/2)/(2*(l+pm)+1))**.5
  d = -pm*((l+pm*(m+1)+1/2)/(2*(l+pm)+1))**.5
  # remove Condon-Shortley phase
  Ya = sph_harm(m-1/2, l, phi, th)*(-1)**(m-1/2) if a != 0 else 0
  Yb = sph_harm(m+1/2, l, phi, th)*(-1)**(m+1/2) if b != 0 else 0
  Yc = sph_harm(m-1/2, l+pm, phi, th)*(-1)**(m-1/2) if c != 0 else 0
  Yd = sph_harm(m+1/2, l+pm, phi, th)*(-1)**(m+1/2) if d != 0 else 0
  jp = j+1/2
  Jp = (jp**2-alpha**2)**.5
  N = (n**2-2*(n-jp)*(jp-Jp))**.5
  eps = (1-(alpha/N)**2)**.5
  f = math.factorial
  G = math.gamma
  R_norm = (G(2*Jp-jp+n+1)/(4*N*(N+pm*jp)*G(2*Jp+1)**2*f(round(n-jp)))*(2/N)**3)**.5
  rho = 2*xi/N
  M0 = hyp1f1(-n+jp,2*Jp+1,rho)
  M1 = hyp1f1(-n+jp+1,2*Jp+1,rho)
  R_phi = R_norm*(1+eps)**.5*rho**(Jp-1)*np.exp(-rho/2)*((jp-n)*M1+(N+pm*jp)*M0)
  R_chi = 1j*R_norm*(1-eps)**.5*rho**(Jp-1)*np.exp(-rho/2)*((jp-n)*M1-(N+pm*jp)*M0)
  return (a*R_phi*Ya, b*R_phi*Yb, c*R_chi*Yc, d*R_chi*Yd)

def s(psi, rho) -> tuple[float,float,float]:
  ss = N//N_quiver
  red = lambda x: x[ss//2::ss,ss//2::ss]
  a, b, c, d = psi
  rho, a, b, c, d = red(rho), red(a), red(b), red(c), red(d)
  C = lambda x: x.conjugate()
  return (((C(a)*b+C(b)*a+C(c)*d+C(d)*c)/rho).real,
          ((-1j*C(a)*b+1j*C(b)*a-1j*C(c)*d+1j*C(d)*c)/rho).real,
          ((C(a)*a-C(b)*b+C(c)*c-C(d)*d)/rho).real)

def eval_rho_volume() -> np.ndarray:
  xyz = np.meshgrid(ls := np.linspace(-hr, hr, N), ls, ls)
  psis = [tuple(c*psi_ for psi_ in psi(*nljm, alpha, *C_to_s(*xyz))) for c, nljm in state]
  psi_ = tuple(sum(e) for e in zip(*psis))
  rho = sum(abs(e)**2 for e in psi_)
  return rho

def eval_rho_plane(plane:str, slice_, alpha) -> np.ndarray:
  a, b = np.meshgrid(ls := np.linspace(-hr, hr, N), ls)
  match plane:
    case 'xy': xyz = a, b, slice_
    case 'xz': xyz = a, slice_, b
    case 'yz': xyz = slice_, a, b
  psis = [tuple(c*psi_ for psi_ in psi(*nljm, alpha, *C_to_s(*xyz))) for c, nljm in state]
  psi_ = tuple(sum(e) for e in zip(*psis))
  rho = sum(abs(e)**2 for e in psi_)
  return (rho, s(psi_, rho), (a, b)) if quiver_2D else rho

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
  def plot(data, plane, slice, alpha):
    if quiver_2D: data, s_, mg = data
    if sqrt_scaling: data **= .5
    nonlocal vmax
    if vmax is None: vmax = np.max(data)
    ax.clear()
    ax.set_xlabel(f'${plane[0]}/a$'); ax.set_ylabel(f'${plane[1]}/a$')
    title = f'${(set('xyz')-set(plane)).pop()}/a={slice:.1f}$ $\\alpha={alpha:.2f}$'
    if len(state) == 1 and plot_title:
      _, (n, l, j, m) = state[0]
      title = f'$(n,l,j,m)=({n},{l},{j},{m})$\n{title}'
    ax.set_title(title)
    img = ax.imshow(data, extent=(-hr, hr, -hr, hr), origin='lower', vmin=0, vmax=vmax, aspect='equal',
      cmap='cividis', interpolation='none')
    if quiver_2D:
      ss = N//N_quiver
      red = lambda x: x[ss//2::ss,ss//2::ss]
      (sx, sy, sz), (a, b) = s_, mg
      a, b = red(a), red(b)
      match plane:
        case 'xy': s1, s2 = sx, sy
        case 'xz': s1, s2 = sx, sz
        case 'yz': s1, s2 = sy, sz
      ax.quiver(a, b, s1, s2, angles='xy')
    return img
  data = eval_rho_plane(plane_init, slice_init, alpha)
  img = plot(data, plane_init, slice_init, alpha)
  fig.colorbar(img, ax=ax, fraction=.046, pad=.04).set_label(R'$\sqrt{\rho}$' if sqrt_scaling else R'$\rho$')
  ax.set_box_aspect(1)
  plt.tight_layout()
  if not plot_save:
    slice_slider_ax, alpha_slider_ax, plane_radio_ax = plt.axes([.3, 0, .55, .03]), plt.axes([.3, .05, .55, .03]), plt.axes([0, 0, .15, .15])
    slice_slider = Slider(slice_slider_ax, '', -hr, hr, valinit=slice_init, dragging=not use_tex, valfmt='%.2f')
    alpha_slider = Slider(alpha_slider_ax, '', 0, 1, valinit=alpha, dragging=not use_tex, valfmt='%.2f')
    slice_slider.valtext.set_visible(False)
    alpha_slider.valtext.set_visible(False)
    plane_radio = RadioButtons(plane_radio_ax, planes)
    def update(_):
      plane, slice, alpha = plane_radio.value_selected, slice_slider.val, alpha_slider.val
      data = eval_rho_plane(plane, slice, alpha)
      plot(data, plane, slice, alpha)
      fig.canvas.draw_idle()
    slice_slider.on_changed(update)
    alpha_slider.on_changed(update)
    plane_radio.on_clicked(update)
  if plot_save:
    plt.savefig(Path('out')/'H_Dirac_sol.pdf', bbox_inches='tight')
  else: plt.show()

def run_plot_3D():
  import vispy.app, vispy.scene
  canvas = vispy.scene.SceneCanvas('', keys='interactive', show=True)
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
  data = eval_rho_volume().astype(np.float32)**(.5 if sqrt_scaling else 1)
  volume = vispy.scene.visuals.Volume(data, parent=view.scene)
  volume.cmap = 'magma'
  volume.method = 'mip'
  view.camera.set_range()
  canvas.app.run()

if __name__ == '__main__':
  for (_, (_, l, j, _)) in state: assert abs(j-l) == 1/2
  state = [(c/sum(abs(c)**2 for c, _ in state)**.5, nlm) for c, nlm in state]
  run_plot_3D() if plot_3D else run_plot_2D()
