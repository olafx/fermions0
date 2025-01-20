from typing import Callable
from itertools import pairwise
import colorsys
import numpy as np
import matplotlib.pyplot as plt

C_to_s = lambda x, y, z: (r := (x**2+y**2+z**2)**.5, np.arccos(z/r), np.arctan2(y, x))
s_to_C = lambda r, th, phi: (r*np.sin(th)*np.cos(phi), r*np.sin(th)*np.sin(phi), r*np.cos(th))

def C_to_s(x, y, z):
  r = np.sqrt(x**2+y**2+z**2)
  return r, np.arccos(z/r), np.arctan2(y, x)

def almost_equal(a, b, rel_tol=1e-3) -> bool:
  return abs((a-b)/a) < rel_tol

def equal_span_partition(data:np.ndarray, n:int) -> list[tuple]:
  return list(pairwise([int(np.argmax(data >= data[0]+data[-1]/n*i)) for i in range(n+1)]))

def analyze_path(path:np.ndarray, tol=1e-6) -> tuple[bool,float|None]:
  circular = True
  t = path[:,0]
  r, th, phi = C_to_s(path[:,1], path[:,2], path[:,3])
  diff1 = lambda x: (x[2:]-x[:-2])/2
  diff2 = lambda x: x[2:]+x[:-2]-2*x[1:-1]
  difft = lambda x, t: (x[1:]-x[:-1])/(t[1:]-t[:-1])
  for c in (r, th): circular = circular and (diff1(c) < tol).all()
  dphi, d2phi = diff1(phi), diff2(phi)
  filter = np.where(np.abs(d2phi) < tol)
  mad = np.mean(np.abs(dphi[filter]-np.mean(dphi[filter])))
  circular = circular and mad < tol
  return circular, np.mean(difft(phi, t)[filter]) if circular else None

def validate_circular_paths(paths:list[np.ndarray], samples:np.ndarray, expected_vel:Callable):
  for i, (path, sample) in enumerate(zip(paths, samples)):
    circular, dphidt = analyze_path(path)
    assert circular and almost_equal(dphidt, expected_vel(C_to_s(*sample)[0])), f'error in path {i}'

def dphidtau(path:np.ndarray, time_scale:float, filter_under:float) -> tuple[np.ndarray,np.ndarray]:
  tau = path[:,0]/time_scale
  phi = C_to_s(*path[:,1:].T)[2]
  dphidtau = (phi[1:]-phi[:-1])/(tau[1:]-tau[:-1])
  filter = np.where(np.abs(dphidtau) < filter_under)
  return filter, dphidtau

def colors_and_shades(n_colors, n_shades, lightness_range):
  def darken(rgb, lightness):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l*lightness), s)
  return [[darken(bc, l) for l in np.linspace(*lightness_range, n_shades)] for bc in plt.cm.tab10.colors[:n_colors]]

def filter_r_max(samples, r_max):
  return samples[(samples[:,0]**2 <= r_max**2)*(samples[:,1]**2 <= r_max**2)*(samples[:,2]**2 <= r_max**2)]
