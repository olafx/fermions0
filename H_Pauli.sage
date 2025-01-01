n, l, j, m = 2, 1, 3/2, +1/2
n_ = 1 # the effective n used in undimensionalizing

from sage.manifolds.operators import *
from sage.manifolds.utilities import simplify_chain_real
E.<xi,th,ph> = EuclideanSpace(coordinates='spherical')
var('l_ m_ x y z rho chi M')

def nice(expr):
  # spherical coordinate constraints
  # explicit confluent hypergeometric function
  # power based trigonometric simplification
  # general simplification
  # factoring
  return simplify_chain_real(expr).hypergeometric_simplify().trig_simplify().full_simplify().factor()

def nice_vector_field(V):
  # factoring
  return V.parent()([V[1].factor(), V[2].factor(), V[3].factor()])

p = j-l == 1/2
pm = 2*p-1
a = sqrt((l+pm*m+1/2)/(2*l+1))
b = -pm*sqrt((l-pm*m+1/2)/(2*l+1))

# basic functions
f = factorial
P_(l_,m_,x) = gen_legendre_P(l_,m_,x)*(-1)^m_
P(l_,m_,x) = sqrt((2*l_+1)/2*f(l_-m_)/f(l_+m_))*P_(l_,m_,x)
Y(l_,m_,th,ph) = 1/sqrt(2*pi)*P(l_,m_,cos(th))*exp(I*m_*ph)
Pa(th) = nice(P(l,m-1/2,cos(th))) if a != 0 else 0
Pb(th) = nice(P(l,m+1/2,cos(th))) if b != 0 else 0
Ya(th,ph) = Y(l,m-1/2,th,ph) if a != 0 else 0
Yb(th,ph) = Y(l,m+1/2,th,ph) if b != 0 else 0
R_(rho) = exp(-rho/2)*rho^l*hypergeometric([-n+l+1],[2*l+2],rho)
R(rho) = 1/f(2*l+1)*sqrt(f(n+l)/(f(n-l-1)*2*n))*(2/n)^(3/2)*R_(rho)
rho_from_xi(xi) = 2*xi/n
psi_a(xi,th,ph) = a*R(rho_from_xi(xi))*Ya(th,ph)
psi_b(xi,th,ph) = b*R(rho_from_xi(xi))*Yb(th,ph)
rho_(xi,th,ph) = R(rho_from_xi(xi))^2/(2*pi)*(a^2*Pa(th)^2+b^2*Pb(th)^2)
s_xi(th) = (2*a*b*Pa(th)*Pb(th)*sin(th)+(a^2*Pa(th)^2-b^2*Pb(th)^2)*cos(th))/(a^2*Pa(th)^2+b^2*Pb(th)^2)
s_th(th) = (2*a*b*Pa(th)*Pb(th)*cos(th)-(a^2*Pa(th)^2-b^2*Pb(th)^2)*sin(th))/(a^2*Pa(th)^2+b^2*Pb(th)^2)
s_z(th) = (a^2*Pa(th)^2-b^2*Pb(th)^2)/(a^2*Pa(th)^2+b^2*Pb(th)^2)
dPa_dth(th) = derivative(Pa(th),th)
dPb_dth(th) = derivative(Pb(th),th)
dlogR_drho(xi) = derivative(log(R(rho_from_xi(xi))),xi)/derivative(rho_from_xi(xi),xi)

# vector fields and associated results
s_s = E.vector_field(s_xi(th), s_th(th), 0)
rho_s = E.scalar_field(nice(rho_(xi,th,ph)))
v1_s = E.vector_field(0, 0, nice(2/(xi*sin(th))*(m-1/2*s_z(th))))
v2_s = curl(rho_s*s_s)/rho_s
v3_s = M*chi*s_s
dX_dtau_s = nice_vector_field(n_^2*(v1_s+v2_s+v3_s))
dxi_dtau = dX_dtau_s[1].expr()
dth_dtau = (dX_dtau_s[2]/xi).expr()
dph_dtau = (dX_dtau_s[3]/(xi*sin(th))).expr()
r = (-n_^2*M*chi*div(rho_s*s_s)/rho_s).expr()

print(f'a\n{a}')
print(f'b\n{b}')
print(f'rho\n{nice(rho_(xi,th,ph))}')
print(f's_xi\n{nice(s_xi)}')
print(f's_th\n{nice(s_th)}')
print(f'v1\n{nice_vector_field(v1_s).display()}')
print(f'v2\n{nice_vector_field(v2_s).display()}')
print(f'v3\n{nice_vector_field(v3_s).display()}')
print(f'd xi / d tau\n{nice(dxi_dtau)}')
print(f'd th / d tau\n{nice(dth_dtau)}')
print(f'd ph / d tau\n{nice(dph_dtau)}')
print(f'r\n({nice(r)})^+')

# verification of key analytical results
a1 = nice_vector_field(curl(rho_s*s_s)/rho_s)
a2 = nice_vector_field(E.vector_field(0, 0, 4/n*dlogR_drho(xi)*s_th(th)
  -2/xi/(a^2*Pa(th)^2+b^2*Pb(th)^2)
  *(a*b*sin(th)*(Pa(th)*dPb_dth(th)+Pb(th)*dPa_dth(th))
  +a^2*cos(th)*Pa(th)*dPa_dth(th)
  -b^2*cos(th)*Pb(th)*dPb_dth(th))
  ))
b1 = nice((div(rho_s*s_s)/rho_s).expr())
b2 = nice(4/n*dlogR_drho(xi)*s_xi(th)
  +2/xi/(a^2*Pa(th)^2+b^2*Pb(th)^2)
  *(a*b*(cos(th)*(Pa(th)*dPb_dth(th)+Pb(th)*dPa_dth(th))+Pa(th)*Pb(th)/sin(th))
  -a^2*sin(th)*Pa(th)*dPa_dth(th)
  +b^2*sin(th)*Pb(th)*dPb_dth(th)))
assert a1 == a2
assert b1 == b2
