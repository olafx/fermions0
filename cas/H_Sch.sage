n, l, m = 1, 0, 0
n_ = 1 # the effective n used in undimensionalizing

from sage.manifolds.operators import *
from sage.manifolds.utilities import simplify_chain_real
E.<xi,th,ph> = EuclideanSpace(coordinates='spherical')
var('x y z rho chi M')

def nice(expr):
  # spherical coordinate constraints
  # explicit confluent hypergeometric function
  # power based trigonometric simplification
  # general simplification
  # factoring
  return simplify_chain_real(expr).hypergeometric_simplify().trig_simplify().full_simplify().factor()

def nice_vector_field(V):
  return V.parent()([V[1].factor(), V[2].factor(), V[3].factor()])

# basic functions
f = factorial
P_(x) = gen_legendre_P(l,m,x)*(-1)^m
P(x) = sqrt((2*l+1)/2*f(l-m)/f(l+m))*P_(x)
Y(th,ph) = 1/sqrt(2*pi)*P(cos(th))*exp(I*m*ph)
R_(rho) = exp(-rho/2)*rho^l*hypergeometric([-n+l+1],[2*l+2],rho)
R(rho) = 1/f(2*l+1)*sqrt(f(n+l)/(f(n-l-1)*2*n))*(2/n)^(3/2)*R_(rho)
rho_from_xi(xi) = 2*xi/n
psi(xi,th,ph) = R(rho_from_xi(xi))*Y(th,ph)
S = arg(psi(xi,th,ph))
rho_(xi,th,ph) = conjugate(psi(xi,th,ph))*psi(xi,th,ph)
dlogP_dth(th) = derivative(log(P(cos(th))),th)
dlogR_drho(xi) = derivative(log(R(rho_from_xi(xi))),xi)/derivative(rho_from_xi(xi),xi)

# vector fields and associated results
s_s = E.vector_field(0, 0, 1, frame=E.cartesian_frame())
psi_s = E.scalar_field(nice(psi(xi,th,ph)))
rho_s = E.scalar_field(nice(rho_(xi,th,ph)))
S_s = E.scalar_field(nice(S))
dX_dtau_s = nice_vector_field(n_^2*(2*grad(S_s)+grad(log(rho_s)).cross_product(s_s))+M*chi*s_s)
dxi_dtau = dX_dtau_s[1].expr()
dth_dtau = (dX_dtau_s[2]/xi).expr()
dph_dtau = (dX_dtau_s[3]/(xi*sin(th))).expr()
r = (-n_^2*M*chi*grad(log(rho_s)).dot(s_s)).expr()

print(f'_1F_1({-n+l+1};{2*l+2};rho)\n{nice(hypergeometric([-n+l+1],[2*l+2],rho))}')
print(f'_1F_1({-n+l+1};{2*l+3};rho)\n{nice(hypergeometric([-n+l+1],[2*l+3],rho))}')
print(f'R\n{nice(R_(rho_from_xi(xi)))}')
print(f'cal R\n{nice(R(rho_from_xi(xi)))}')
print(f'P\n{nice(P_(cos(th)))}')
print(f'cal P\n{nice(P(cos(th)))}')
print(f'Y\n{nice(Y(th,ph))}')
print(f'psi\n{nice(psi(xi,th,ph))}')
print(f'rho\n{nice(rho_(xi,th,ph))}')
print(f'S = {nice(S)}')
print(f'd log cal P / d th\n{nice(dlogP_dth(th))}')
print(f'd log cal R / d rho\n{nice(dlogR_drho(xi))}')
print(f's\n{s_s.display()}')
print(f'd X / d tau\n{dX_dtau_s.display()}')
print(f'd xi / d tau\n{nice(dxi_dtau)}')
print(f'd th / d tau\n{nice(dth_dtau)}')
print(f'd ph / d tau\n{nice(dph_dtau)}')
print(f'r\n({nice(r)})^+')

# verification of key analytical results
# a1 = nice_vector_field(grad(log(rho_s)).cross_product(s_s))
# a2 = nice_vector_field(E.vector_field(0, 0, nice(-4/n*sin(th)*dlogR_drho(xi)-2/xi*cos(th)*dlogP_dth(th))))
# b1 = nice(grad(log(rho_s)).dot(s_s).expr())
# b2 = nice(4/n*cos(th)*dlogR_drho(xi)-2/xi*sin(th)*dlogP_dth(th))
# assert a1 == a2
# assert b1 == b2
