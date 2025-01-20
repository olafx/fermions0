import numpy as np
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from util import colors_and_shades

# physics params
# (n,l,j) eigenstates
states = [(1, 0, 1/2), (2, 0, 1/2), (2, 1, 1/2), (2, 1, 3/2)] # blue, orange, green, red
alphas = [.5, 0.999] # fine structure constants
alphas_fmt = ['--', '']
xi_range = (.2, 10)
# plot params
ylim_phi = (1e-3, 1e1)
ylim_chi = (1e-4, 1e1)
# ylim_phi = None
# ylim_chi = None
N = 10000 # resolution
plot_log = True
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
plot_save = True # save 2D plot instead of showing

def R(n, l, j, alpha, xi) -> tuple[float,float]:
  p = j-l == 1/2
  pm = 2*p-1
  jp = j+1/2
  Jp = (jp**2-alpha**2)**.5
  N = (n**2-2*(n-jp)*(jp-Jp))**.5
  eps = (1-(alpha/N)**2)**.5
  rho = 2*xi/N
  M0 = hyp1f1(-n+jp,2*Jp+1,rho) if N+pm*jp != 0 else 0
  M1 = hyp1f1(-n+jp+1,2*Jp+1,rho) if jp-n != 0 else 0
  R_phi = (1+eps)**.5*rho**(Jp-1)*np.exp(-rho/2)*((jp-n)*M1+(N+pm*jp)*M0)
  R_chi = 1j*(1-eps)**.5*rho**(Jp-1)*np.exp(-rho/2)*((jp-n)*M1-(N+pm*jp)*M0)
  return R_phi, R_chi

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (8, 4)
if use_tex: plt.rcParams['text.usetex'] = True
if plot_dark: plt.style.use('dark_background')
if plot_save:
  from pathlib import Path
  from os import makedirs
  makedirs('out', exist_ok=True)
plt.figure('Dirac radial')
ax1, ax2 = plt.subplot(121), plt.subplot(122)
colors = colors_and_shades(len(states), len(alphas), (1, 1))
for i, nlj in enumerate(states):
  for j, (alpha, fmt) in enumerate(zip(alphas, alphas_fmt)):
    R_phi, R_chi = R(*nlj, alpha, xi := np.linspace(*xi_range, N))
    ax1.plot(xi, np.abs(R_phi), fmt, c=colors[i][j])
    ax2.plot(xi, np.abs(R_chi), fmt, c=colors[i][j])
ax1.set_ylabel(R'$|\mathcal{R}_\pm^\varphi|/\mathcal{R}_\pm^0$')
ax2.set_ylabel(R'$|\mathcal{R}_\pm^\chi|/\mathcal{R}_\pm^0$')
if plot_log: ax1.set_yscale('log'); ax2.set_yscale('log')
ax1.set_xscale('log'); ax2.set_xscale('log')
ax1.set_xlim(*xi_range); ax2.set_xlim(*xi_range)
ax1.set_xlabel(R'$\xi$'); ax2.set_xlabel(R'$\xi$')
if ylim_phi is not None: ax1.set_ylim(*ylim_phi)
if ylim_chi is not None: ax2.set_ylim(*ylim_chi)
plt.tight_layout()
if plot_save: plt.savefig(Path('out')/'H_Dirac_radial.pdf', bbox_inches='tight')
if not plot_save: plt.show()
