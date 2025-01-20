from sage.all import *
import numpy as np
from scipy.optimize import least_squares

ones2 = identity_matrix(2)

sig1 = Matrix([[0, 1], [1, 0]])
sig2 = Matrix([[0, -I], [I, 0]])
sig3 = Matrix([[+1, 0], [0, -1]])

gammaW = [block_matrix([[0, ones2], [ones2, 0]]),
          block_matrix([[0, sig1], [-sig1, 0]]),
          block_matrix([[0, sig2], [-sig2, 0]]),
          block_matrix([[0, sig3], [-sig3, 0]])]
gammaD = [block_matrix([[ones2, 0], [0, -ones2]]),
          gammaW[1],
          gammaW[2],
          gammaW[3]]

# sol in 1 -> sol in 2
gamma1 = gammaD
gamma2 = gammaW

gamma1_ = [np.array(m, dtype=np.complex128) for m in gamma1]
gamma2_ = [np.array(m, dtype=np.complex128) for m in gamma2]

def residuals(params):
  U_12 = np.array(params[:4*4]).reshape((4, 4))\
     +1j*np.array(params[4*4:]).reshape((4, 4))
  res = []
  eye1 = np.conjugate(U_12.T)@U_12
  eye2 = U_12@np.conjugate(U_12.T)
  res.extend((eye1-np.eye(4)).flatten().real)
  res.extend((eye1-np.eye(4)).flatten().imag)
  res.extend((eye2-np.eye(4)).flatten().real)
  res.extend((eye2-np.eye(4)).flatten().imag)
  for i in range(4):
    gamma2_trial = np.conjugate(U_12.T)@gamma1_[i]@U_12
    res.extend((gamma2_trial-gamma2_[i]).flatten().real)
    res.extend((gamma2_trial-gamma2_[i]).flatten().imag)
  return res

initial_guess = np.random.rand(4*4*2)
res = least_squares(residuals, initial_guess, method='lm')
U_12 = res.x[:4*4].reshape((4, 4))+1j*res.x[4*4:].reshape((4, 4))

print(f'U_12\n{U_12}')

assert np.allclose(np.conjugate(U_12.T)@U_12, np.eye(4))
assert np.allclose(U_12@np.conjugate(U_12.T), np.eye(4))
for i in range(4):
  assert np.allclose(np.conjugate(U_12.T)@gamma1_[i]@U_12, gamma2_[i])
