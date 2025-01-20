n, l, j, m = 1, 0, 1/2, +1/2
n_ = 1 # the effective n used in undimensionalizing

from sage.manifolds.operators import *
from sage.manifolds.utilities import simplify_chain_real
E.<xi,th,ph> = EuclideanSpace(coordinates='spherical')
var('l_ m_ x y z rho chi M al')

assume(al, 'real')
assume(0 <= al)
assume(al <= 1)

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
Jp(al) = sqrt(jp^2-al^2)
N(al) = sqrt(n^2-2*(n-jp)*(jp-Jp(al)))
eps(al) = sqrt(1-al^2/N(al)^2)

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
Rphi_(al,rho) =   sqrt(1+eps(al))*rho^(Jp(al)-1)*exp(-rho/2)*\
  (-(n-jp)       *hypergeometric([-n+jp+1],[2*Jp(al)+1],rho)
   +(N(al)+pm*jp)*hypergeometric([-n+jp  ],[2*Jp(al)+1],rho))
Rchi_(al,rho) = I*sqrt(1-eps(al))*rho^(Jp(al)-1)*exp(-rho/2)*\
  (-(n-jp)       *hypergeometric([-n+jp+1],[2*Jp(al)+1],rho)
   -(N(al)+pm*jp)*hypergeometric([-n+jp  ],[2*Jp(al)+1],rho))
Rn(al) = sqrt(g(2*Jp(al)-jp+n+1)/(4*N(al)*(N(al)+pm*jp)*g(2*Jp(al)+1)^2*f(n-jp)))*(2/N(al))^(3/2)
Rphi(al,rho) = Rn(al)*Rphi_(al,rho)
Rchi(al,rho) = Rn(al)*Rchi_(al,rho)
rho_from_xi(al,xi) = 2*xi/N(al)
absRphi(al,xi) = abs(Rphi(al,rho_from_xi(al,xi)))
absRchi(al,xi) = abs(Rchi(al,rho_from_xi(al,xi)))
# Certain solutions don't like simplifying these psi_a, etc., can comment them out.
# psi_a(al,xi,th,ph) = nice(a*Rphi(al,rho_from_xi(al,xi))*Ya(th,ph))
# psi_b(al,xi,th,ph) = nice(b*Rphi(al,rho_from_xi(al,xi))*Yb(th,ph))
# psi_c(al,xi,th,ph) = nice(c*Rchi(al,rho_from_xi(al,xi))*Yc(th,ph))
# psi_d(al,xi,th,ph) = nice(d*Rchi(al,rho_from_xi(al,xi))*Yd(th,ph))
rho_(al,xi,th,ph) = (absRphi(al,xi)^2*(a^2*Pa(th)^2+b^2*Pb(th)^2)
                    +absRchi(al,xi)^2*(c^2*Pc(th)^2+d^2*Pd(th)^2))/(2*pi)
z_s(al,xi,th) = (absRphi(al,xi)^2*(a^2*Pa(th)^2-b^2*Pb(th)^2)
                +absRchi(al,xi)^2*(c^2*Pc(th)^2-d^2*Pd(th)^2))/(2*pi*rho_(al,xi,th,ph))
s_xi(al,xi,th) = sqrt(1-z_s(al,xi,th)^2)*sin(th)+z_s(al,xi,th)*cos(th)
s_th(al,xi,th) = sqrt(1-z_s(al,xi,th)^2)*cos(th)-z_s(al,xi,th)*sin(th)
s5n(al,xi,th,ph) = 2*abs(Rphi(al,rho_from_xi(al,xi))*(-I*Rchi(al,rho_from_xi(al,xi)))*(a*d*Pa(th)*Pd(th)-b*c*Pb(th)*Pc(th)))/(2*pi*rho_(al,xi,th,ph))
sn(al,xi,th,ph) = sqrt(1-s5n(al,xi,th,ph)^2)
s5_ph(al,xi,th,ph) = 2*Rphi(al,rho_from_xi(al,xi))*(-I*Rchi(al,rho_from_xi(al,xi)))*(a*d*Pa(th)*Pd(th)-b*c*Pb(th)*Pc(th))/(2*pi*rho_(al,xi,th,ph))
r(al,M,xi,th,ph) = M^2*chi*(2*Rphi(al,rho_from_xi(al,xi))*(-I*Rchi(al,rho_from_xi(al,xi)))*(a*c*Pa(th)*Pc(th)+b*d*Pb(th)*Pd(th)))/(2*pi*rho_(al,xi,th,ph))

print(f'a\n{a}')
print(f'b\n{b}')
print(f'c\n{c}')
print(f'd\n{d}')
print(f'Pa\n{Pa}')
print(f'Pb\n{Pb}')
print(f'Pc\n{Pc}')
print(f'Pd\n{Pd}')
print(f'R0\n{nice(Rn)}')
print(f'J_+\n{nice(Jp(al))}')
print(f'N\n{nice(N(al))}')
print(f'rho\n{nice(rho_(al,xi,th,ph))}')
print(f'z_s\n{nice(z_s)}')
print(f's_xi\n{nice(s_xi)}')
print(f's_th\n{nice(s_th)}')
print(f'|s_5|\n{nice(s5n)}')
print(f'|s|\n{nice(sn)}')
print(f'(s_5)_ph\n{nice(s5_ph)}')
print(f'r\n({nice(r)})^+')

# verification of key analytical results

# psi1(al,xi,th,ph) = (psi_a(al,xi,th,ph)-psi_c(al,xi,th,ph))/sqrt(2)
# psi2(al,xi,th,ph) = (psi_b(al,xi,th,ph)-psi_d(al,xi,th,ph))/sqrt(2)
# psi3(al,xi,th,ph) = (psi_a(al,xi,th,ph)+psi_c(al,xi,th,ph))/sqrt(2)
# psi4(al,xi,th,ph) = (psi_b(al,xi,th,ph)+psi_d(al,xi,th,ph))/sqrt(2)
# rho1(al,xi,th,ph) = abs(psi_a(al,xi,th,ph))^2\
#                    +abs(psi_b(al,xi,th,ph))^2\
#                    +abs(psi_c(al,xi,th,ph))^2\
#                    +abs(psi_d(al,xi,th,ph))^2
# assert nice(rho1) == nice(rho_)

# s_x(al,xi,th,ph) = (C(psi_a(al,xi,th,ph))*psi_b(al,xi,th,ph)+C(psi_b(al,xi,th,ph))*psi_a(al,xi,th,ph)\
#                    +C(psi_c(al,xi,th,ph))*psi_d(al,xi,th,ph)+C(psi_d(al,xi,th,ph))*psi_c(al,xi,th,ph))/rho_(al,xi,th,ph)
# s_y(al,xi,th,ph) = (-I*C(psi_a(al,xi,th,ph))*psi_b(al,xi,th,ph)+I*C(psi_b(al,xi,th,ph))*psi_a(al,xi,th,ph)\
#                     -I*C(psi_c(al,xi,th,ph))*psi_d(al,xi,th,ph)+I*C(psi_d(al,xi,th,ph))*psi_c(al,xi,th,ph))/rho_(al,xi,th,ph)
# s_z(al,xi,th,ph) = (abs(psi_a(al,xi,th,ph))^2-abs(psi_b(al,xi,th,ph))^2\
#                    +abs(psi_c(al,xi,th,ph))^2-abs(psi_d(al,xi,th,ph))^2)/rho_(al,xi,th,ph)
# sn2(al,xi,th,ph) = sqrt(s_x(al,xi,th,ph)^2+s_y(al,xi,th,ph)^2+s_z(al,xi,th,ph)^2)
# s5n2(al,xi,th,ph) = abs(2*(C(psi_a(al,xi,th,ph))*C(psi_d(al,xi,th,ph))-C(psi_b(al,xi,th,ph))*C(psi_c(al,xi,th,ph))))/rho_(al,xi,th,ph)
# assert nice(s5n) == nice(s5n2)
# assert nice(sn) == nice(sn2) # true but fails to validate
# p1 = plot(nice(sn(  al=x ,xi=1,th=1,ph=1)^2), (x,0,1));  p1.save('p1.png')
# p2 = plot(nice(sn2( al=x ,xi=1,th=1,ph=1)^2), (x,0,1));  p2.save('p2.png')
# p3 = plot(nice(s5n( al=x ,xi=1,th=1,ph=1)^2), (x,0,1));  p3.save('p3.png')
# p4 = plot(nice(s5n2(al=x ,xi=1,th=1,ph=1)^2), (x,0,1));  p4.save('p4.png')
# p5 = plot(nice(sn(  al=.5,xi=1,th=x,ph=1)^2), (x,0,pi)); p5.save('p5.png')
# p6 = plot(nice(sn2( al=.5,xi=1,th=x,ph=1)^2), (x,0,pi)); p6.save('p6.png')
# p7 = plot(nice(s5n( al=.5,xi=1,th=x,ph=1)^2), (x,0,pi)); p7.save('p7.png')
# p8 = plot(nice(s5n2(al=.5,xi=1,th=x,ph=1)^2), (x,0,pi)); p8.save('p8.png')

# b1(al,th) = (9*(Jp(al)-2)*sin(th)^2-8*Jp(al)+26)*sin(th)^2-8
# b2(al,th) = (9*(2-Jp(al))*cos(th)^2+10*(Jp(al)-1))*cos(th)^2-Jp(al)
# assert nice(b1) == nice(-b2)
