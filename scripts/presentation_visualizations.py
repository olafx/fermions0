import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from pathlib import Path
import os

t_f = 3e0
r = 1.2e2
s_5_norm = .9e1
t_vid = 15
fps = 60
N_q = 24
x_range = (-1,+1)
x0 = [-1,0]
chi0 = +1
plot = True
vid = False
dpi = 400

# some normalization
r /= t_vid
s_5_norm /= t_vid

seed = 2036-8-12+12
rng = np.random.default_rng(seed)

C_to_p = lambda x, y: ((x**2+y**2)**.5, np.atan2(y, x))
p_to_C = lambda r, th: (r*np.cos(th), r*np.sin(th))
mg = np.meshgrid(x := np.linspace(*x_range, N_q), x)
mg_small = np.meshgrid(x := np.linspace(*x_range, N_q//2), x)

# dipole
s = lambda x, y: (3*x*y/(r := C_to_p(x, y)[0])**2, (2*y**2-x**2)/r**2)
m_s = lambda x, y: tuple(-a for a in s(x, y))
# typical drift from slice away from origin
s_5 = lambda x, y: (s_5_norm*np.cos(x), 0)
# left/right fields
p_s_p = lambda x, y: tuple(+a+b for a, b in zip(s(x, y), s_5(x, y)))
m_s_m = lambda x, y: tuple(-a+b for a, b in zip(s(x, y), s_5(x, y)))

plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True

N_t = t_vid*fps
dt = t_f/N_t
path_secs = [[x0+[chi0]]]
path_full = [x0+[chi0]]
x = list(x0)
chi = chi0
t = 0
for i in range(N_t-1):
  path = path_secs[-1]
  v = (p_s_p if chi == +1 else m_s_m)(*x)
  x[0] += v[0]*dt
  x[1] += v[1]*dt
  path += [x.copy()+[chi]]
  path_full += [x.copy()+[chi]]
  if rng.uniform() < r*dt:
    chi *= -1
    path_secs += [[]]

def plot_path(ax, i_end=N_t):
  i_tot = 0
  for i, path in enumerate(path_secs):
    stop_here = i_tot+len(path) > i_end
    i_stop = i_end-i_tot if stop_here else len(path)
    chi = chi0*(2*(i%2)-1)
    path_ = np.array(path)
    ax.plot(path_[:i_stop,0], path_[:i_stop,1], '-', c='red' if chi == +1 else 'green')
    i_tot += len(path)
    if stop_here: break

if plot:

  plt.figure(1, figsize=(10,5))
  ax1 = plt.subplot(121)
  ax2 = plt.subplot(122)
  ax1.set_title(R'$\mathbf{s}=(\mathbf{s}_++\mathbf{s}_-)/2$')
  ax2.set_title(R'$\mathbf{s}_5=(\mathbf{s}_+-\mathbf{s}_-)/2$')
  plt.figure(2, figsize=(10,5))
  ax3 = plt.subplot(121)
  ax4 = plt.subplot(122)
  ax3.set_title(R'$\mathbf v_+=+\mathbf{s_+}$')
  ax4.set_title(R'$\mathbf v_-=-\mathbf{s_-}$')

  for ax in (ax1,ax2,ax3,ax4):
    ax.set_xlabel('$x$')
    ax.set_ylabel('$z$')
    # ax.set_xlim(*x_range)
    # ax.set_ylim(*x_range)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

  plot_path(ax3)
  plot_path(ax4)
  ax1.quiver(*mg, *s(*mg))
  ax2.quiver(*mg, *s_5(*mg))
  ax3.quiver(*mg, *p_s_p(*mg))
  ax4.quiver(*mg, *m_s_m(*mg))

  plt.figure(1)
  plt.tight_layout()
  plt.savefig(Path('out')/'1.pdf')
  plt.figure(2)
  plt.tight_layout()
  plt.savefig(Path('out')/'2.pdf')

if vid:

  path_full = np.array(path_full)
  fig = plt.figure(figsize=(9.3,6))
  gs = gridspec.GridSpec(2, 2, width_ratios=[2,1], height_ratios=[1,1])
  ax1 = plt.subplot(gs[:,0])
  ax2 = plt.subplot(gs[0,1])
  ax3 = plt.subplot(gs[1,1])
  def frame(i):
    print(f'{i+1:{len(str(N_t))}}/{N_t}')
    for ax in (ax1,ax2,ax3):
      ax.clear()
    _, _, chi = path_full[i]
    ax1.quiver(*mg, *(m_s_m if chi == +1 else p_s_p)(*mg), alpha=.1)
    ax1.quiver(*mg, *(p_s_p if chi == +1 else m_s_m)(*mg), alpha=.7)
    plot_path(ax1, i_end=i)
    for ax in (ax1,ax2,ax3):
      ax.set_xlabel('$x$')
      ax.set_ylabel('$z$')
      # ax.set_xlim(*x_range)
      # ax.set_ylim(*x_range)
      ax.set_aspect(1)
      ax.set_xticks([])
      ax.set_yticks([])
    ax2.quiver(*mg_small, *(s if chi == +1 else m_s)(*mg_small))
    ax3.quiver(*mg_small, *s_5(*mg_small))
    ax1.set_title(R'$\mathbf v_+=+\mathbf{s_+}=+\mathbf{s}+\mathbf{s}_5$'
      if chi == +1 else R'$\mathbf v_-=-\mathbf{s_-}=-\mathbf{s}+\mathbf{s}_5$')
    ax2.set_title(f'${'+' if chi == +1 else '-'}'+R'\mathbf{s}$')
    ax3.set_title(R'$\mathbf{s}_5$')
    if i == 0:
      plt.tight_layout()

  ani = animation.FuncAnimation(fig, frame, frames=N_t, interval=1000//fps)
  os.makedirs('out', exist_ok=True)
  ani.save(Path('out')/'3.mp4', writer=animation.FFMpegWriter(fps=fps, bitrate=-1), dpi=dpi)
  fig.savefig(Path('out')/'3.pdf', bbox_inches='tight')
