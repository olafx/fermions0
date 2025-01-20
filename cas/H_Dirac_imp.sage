n, l, j, m = 2, 1, 3/2, -3/2
n_ = 1 # the effective n used in undimensionalizing

from sage.manifolds.operators import *
from sage.manifolds.utilities import simplify_chain_real
E.<xi,th,ph> = EuclideanSpace(coordinates='spherical')
var('l_ m_ x y z rho chi M al Jp N')

assume(al, 'real')
assume(0 <= al)
assume(al <= 1)
assume(N >= 0)
assume(Jp >= 0)

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
c = -sqrt((l-pm*(m-1)+1/2)/(2*(l+pm)+1))
d = -pm*sqrt((l+pm*(m+1)+1/2)/(2*(l+pm)+1))
jp = j+1/2
eps(al,N) = sqrt(1-al^2/N^2)

# basic functions
f = factorial
g = gamma
C = conjugate
P_(l_,m_,x) = gen_legendre_P(l_,m_,x)*(-1)^m_
P(l_,m_,x) = sqrt((2*l_+1)/2*f(l_-m_)/f(l_+m_))*P_(l_,m_,x)
Y(l_,m_,th,ph) = 1/sqrt(2*pi)*P(l_,m_,cos(th))*exp(I*m_*ph)
Pa(th) = nice(P(l,m-1/2,cos(th))) if a != 0 else 0
Pb(th) = nice(P(l,m+1/2,cos(th))) if b != 0 else 0
Pc(th) = nice(P(l+pm,m-1/2,cos(th))) if c != 0 else 0
Pd(th) = nice(P(l+pm,m+1/2,cos(th))) if d != 0 else 0
Ya(th,ph) = nice(Y(l,m-1/2,th,ph)) if a != 0 else 0
Yb(th,ph) = nice(Y(l,m+1/2,th,ph)) if b != 0 else 0
Yc(th,ph) = nice(Y(l+pm,m-1/2,th,ph)) if c != 0 else 0
Yd(th,ph) = nice(Y(l+pm,m+1/2,th,ph)) if d != 0 else 0
Rphi_(al,Jp,N,rho) =   sqrt(1+eps(al,N))*rho^(Jp-1)*exp(-rho/2)*\
  (-(n-jp)       *hypergeometric([-n+jp+1],[2*Jp+1],rho)
   +(N+pm*jp)*hypergeometric([-n+jp  ],[2*Jp+1],rho))
Rchi_(al,Jp,N,rho) = I*sqrt(1-eps(al,N))*rho^(Jp-1)*exp(-rho/2)*\
  (-(n-jp)       *hypergeometric([-n+jp+1],[2*Jp+1],rho)
   -(N+pm*jp)*hypergeometric([-n+jp  ],[2*Jp+1],rho))
Rn(al,Jp,N) = sqrt(g(2*Jp-jp+n+1)/(4*N*(N+pm*jp)*g(2*Jp+1)^2*f(n-jp)))*(2/N)^(3/2)
Rphi(al,Jp,N,rho) = Rn(al)*Rphi_(al,Jp,N,rho)
Rchi(al,Jp,N,rho) = Rn(al)*Rchi_(al,Jp,N,rho)
rho_from_xi(al,xi,N) = 2*xi/N
absRphi(al,Jp,N,xi) = abs(Rphi(al,Jp,N,rho_from_xi(al,xi,N)))
absRchi(al,Jp,N,xi) = abs(Rchi(al,Jp,N,rho_from_xi(al,xi,N)))
# Certain solutions don't like simplifying these psi_a, etc., can comment them out.
# psi_a(al,xi,th,ph) = nice(a*Rphi(al,Jp,N,rho_from_xi(al,xi,N))*Ya(th,ph))
# psi_b(al,xi,th,ph) = nice(b*Rphi(al,Jp,N,rho_from_xi(al,xi,N))*Yb(th,ph))
# psi_c(al,xi,th,ph) = nice(c*Rchi(al,Jp,N,rho_from_xi(al,xi,N))*Yc(th,ph))
# psi_d(al,xi,th,ph) = nice(d*Rchi(al,Jp,N,rho_from_xi(al,xi,N))*Yd(th,ph))
rho_(al,xi,th,ph) = (absRphi(al,Jp,N,xi)^2*(a^2*Pa(th)^2+b^2*Pb(th)^2)
                    +absRchi(al,Jp,N,xi)^2*(c^2*Pc(th)^2+d^2*Pd(th)^2))/(2*pi)
s_z(al,xi,th) = (absRphi(al,Jp,N,xi)^2*(a^2*Pa(th)^2-b^2*Pb(th)^2)
                +absRchi(al,Jp,N,xi)^2*(c^2*Pc(th)^2-d^2*Pd(th)^2))/(2*pi*rho_(al,xi,th,ph))
s_xi(al,xi,th) = sqrt(1-s_z(al,xi,th)^2)*sin(th)+s_z(al,xi,th)*cos(th)
s_th(al,xi,th) = sqrt(1-s_z(al,xi,th)^2)*cos(th)-s_z(al,xi,th)*sin(th)
s5n(al,xi,th,ph) = 2*abs(Rphi(al,rho_from_xi(al,xi,N))*(-I*Rchi(al,Jp,N,rho_from_xi(al,xi,N)))*(a*d*Pa(th)*Pd(th)-b*c*Pb(th)*Pc(th)))/(2*pi*rho_(al,xi,th,ph))
sn(al,xi,th,ph) = sqrt(1-s5n(al,xi,th,ph)^2)
sign(al,xi,th,ph) = sgn(Rphi(al,rho_from_xi(al,xi,N))*(-I*Rchi(al,Jp,N,rho_from_xi(al,xi,N)))*(a*d*Pa(th)*Pd(th)-b*c*Pb(th)*Pc(th)))
r(al,xi,th,ph) = pm*(2*Rphi(al,rho_from_xi(al,xi,N))*(-I*Rchi(al,Jp,N,rho_from_xi(al,xi,N)))*(a*c*Pa(th)*Pc(th)+b*d*Pb(th)*Pd(th)))/(2*pi*rho_(al,xi,th,ph))

print(f'a\n{a}')
print(f'b\n{b}')
print(f'c\n{c}')
print(f'd\n{d}')
exit()
print(f'rho\n{nice(rho_(al,xi,th,ph))}')
print(f's_z\n{nice(s_z)}')
print(f'|s_5|\n{nice(s5n)}')
print(f'r\n({nice(r)})^+')
