from sage.all import *

ones2 = identity_matrix(2)
ones4 = identity_matrix(4)

sig1 = Matrix([[0, 1],
               [1, 0]])
sig2 = Matrix([[0, -I],
               [+I, 0]])
sig3 = Matrix([[+1, 0],
               [0, -1]])

gammaW = [block_matrix([[0, ones2], [ones2, 0]]),
          block_matrix([[0, sig1], [-sig1, 0]]),
          block_matrix([[0, sig2], [-sig2, 0]]),
          block_matrix([[0, sig3], [-sig3, 0]])]
gammaD = [block_matrix([[ones2, 0], [0, -ones2]]),
          gammaW[1],
          gammaW[2],
          gammaW[3]]

U_DW = block_matrix([[+ones2, +ones2],
                     [-ones2, +ones2]])/sqrt(2)
U_BD = block_matrix([[sig1, 0],
                     [0, sig1]])

assert U_DW.H*U_DW == ones4
assert U_DW*U_DW.H == ones4
for i in range(4): assert gammaW[i] == U_DW.H*gammaD[i]*U_DW
