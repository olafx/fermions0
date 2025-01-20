import math
import numpy as np
from scipy.special import sph_harm, hyp1f1
import matplotlib.pyplot as plt
from util import colors_and_shades

# physics params
# (n,l,j,m) eigenstates
# states = [(1, 0, 1/2, +1/2), (2, 0, 1/2, +1/2), (2, 1, 1/2, +1/2), (2, 1, 3/2, +1/2), (2, 1, 3/2, +3/2)]
states = [(1, 0, 1/2, +1/2), (2, 1, 3/2, +1/2), (2, 1, 3/2, +3/2)] # blue, orange, green
alphas = [.2, .8] # fine structure constants
alphas_fmt = ['--', '']
xi = 1
phi = 1
# plot params
N = 1000 # resolution
plot_dark = False # plot 2D with dark background, 3D is always dark
use_tex = True # use LaTeX text rendering
plot_save = True # save 2D plot instead of showing

def s_5_phi_and_r(n, l, j, m, alpha, xi, th, phi) -> tuple[float,float,float,float]:
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
  Pa = np.real(Ya*(2*np.pi)**.5*np.exp(-1j*(m-1/2)*phi))
  Pb = np.real(Yb*(2*np.pi)**.5*np.exp(-1j*(m+1/2)*phi))
  Pc = np.real(Yc*(2*np.pi)**.5*np.exp(-1j*(m-1/2)*phi))
  Pd = np.real(Yd*(2*np.pi)**.5*np.exp(-1j*(m+1/2)*phi))
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
  psi = (a*R_phi*Ya, b*R_phi*Yb, c*R_chi*Yc, d*R_chi*Yd)
  rho_ = sum(np.abs(psi[i])**2 for i in range(4))
  s_5_phi = (2*R_phi*np.imag(R_chi))*(a*d*Pa*Pd-b*c*Pb*Pc)/(2*np.pi*rho_)
  r = (2*R_phi*np.imag(R_chi))*(a*c*Pa*Pc+b*d*Pb*Pd)/(2*np.pi*rho_)
  r_pm_to_mp = pm*r*(pm*r>0)
  r_mp_to_pm = -pm*r*(-pm*r>0)
  return s_5_phi, r_pm_to_mp, r_mp_to_pm

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (8, 7)
if use_tex: plt.rcParams['text.usetex'] = True
if plot_dark: plt.style.use('dark_background')
if plot_save:
  from pathlib import Path
  from os import makedirs
  makedirs('out', exist_ok=True)
plt.figure('Dirac radial')
ax1, ax2, ax3, ax4 = plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)
colors = colors_and_shades(len(states), len(alphas), (1, 1))
for i, nljm in enumerate(states):
  for j, (alpha, fmt) in enumerate(zip(alphas, alphas_fmt)):
    s_5_phi, r_pm_to_mp, r_mp_to_pm = s_5_phi_and_r(*nljm, alpha, xi, th := np.arccos(np.linspace(-1, 1, N)), phi)
    ax1.plot(np.cos(th), s_5_phi, fmt, c=colors[i][j])
    ax2.plot(np.cos(th), r_pm_to_mp, fmt, c=colors[i][j])
    ax3.plot(np.cos(th), (1-np.abs(s_5_phi)**2)**.5, fmt, c=colors[i][j])
    ax4.plot(np.cos(th), r_mp_to_pm, fmt, c=colors[i][j])
ax1.set_xlim(-1, +1); ax2.set_xlim(-1, +1); ax3.set_xlim(-1, +1); ax4.set_xlim(-1, +1)
for ax in (ax1, ax2, ax3, ax4): ax.set_xlabel(R'$\cos\theta$')
ax1.set_ylabel(R'$(\mathbf{s}_5)_\phi$'); ax3.set_ylabel(R'$|\mathbf{s}|$')
ax2.set_ylabel(R'$r(\pm\to\mp)/\mathcal{M}^2$'); ax4.set_ylabel(R'$r(\mp\to\pm)/\mathcal{M}^2$')
plt.tight_layout()
if plot_save: plt.savefig(Path('out')/'H_Dirac_dyn.pdf', bbox_inches='tight')
if not plot_save: plt.show()
