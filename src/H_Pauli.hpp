#pragma once
#include <omp.h>
#include <thread>
#include <complex>
#include <vector>
#include <span>
#include "H_Sch.hpp"
#include "math_util.hpp"
#include "random.hpp"

namespace Pauli_Hydrogen
{

enum Orbital
{ _1_0_1o2_p1o2, _2_0_1o2_p1o2, _2_1_1o2_p1o2, _2_1_3o2_p1o2, _2_1_3o2_p3o2
};
using namespace std::numbers;
using namespace std::complex_literals;
namespace S_H = Schrodinger_Hydrogen;
using S_H::Dynamics, S_H::undim, S_H::redim, S_H::s_to_C, S_H::C_to_s, S_H::gen_ic;

template <Orbital orbital>
double rho_0
( std::array<double, 3> Xs
)
{ auto [xi, th, phi] = Xs;
  if constexpr (orbital == _1_0_1o2_p1o2)
    return exp(-2*xi)*inv_pi;
  if constexpr (orbital == _2_0_1o2_p1o2)
    return exp(-xi)*pow(xi-2,2)*inv_pi/32;
  if constexpr (orbital == _2_1_1o2_p1o2)
    return exp(-xi)*pow(xi,2)*inv_pi/96;
  if constexpr (orbital == _2_1_3o2_p1o2)
    return exp(-xi)*pow(xi,2)*(1+3*pow(cos(th),2))*inv_pi/192;
  if constexpr (orbital == _2_1_3o2_p3o2)
    return exp(-xi)*pow(xi*sin(th),2)*inv_pi/64;
}

template <Orbital orbital, Dynamics dynamics, typename ...A>
auto vel(A... a) {}
template <Orbital orbital, Dynamics dynamics, typename ...A>
auto rate(A... a) {}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _1_0_1o2_p1o2)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 2/xi};
  if constexpr (dynamics == Dynamics::Pauli)
    return vel;
  auto s_xi = cos(th);
  auto s_th = -sin(th);
  vel[0] += M*chi*s_xi;
  vel[1] += M*chi/xi*s_th;
  if constexpr (dynamics == Dynamics::zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _1_0_1o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., 2*M*chi*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_0_1o2_p1o2)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., (xi-4)/(xi-2)/xi};
  if constexpr (dynamics == Dynamics::Pauli)
    return vel;
  auto s_xi = cos(th);
  auto s_th = -sin(th);
  vel[0] += M*chi*s_xi;
  vel[1] += M*chi/xi*s_th;
  if constexpr (dynamics == Dynamics::zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_0_1o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*(xi-4)/(xi-2)*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_1o2_p1o2)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., -(xi-6)/pow(xi,2)};
  if constexpr (dynamics == Dynamics::Pauli)
    return vel;
  auto s_xi = cos(th);
  auto s_th = sin(th);
  vel[0] += M*chi*s_xi;
  vel[1] += M*chi/xi*s_th;
  if constexpr (dynamics == Dynamics::zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_1o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*(xi-6)/xi*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p1o2)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., (9*pow(cos(th),2)-1)/(3*pow(cos(th),2)+1)/xi};
  if constexpr (dynamics == Dynamics::Pauli)
    return vel;
  auto s_xi = -(5-9*pow(cos(th),2))*cos(th)/(1+3*pow(cos(th),2));
  auto s_th = +(1-9*pow(cos(th),2))*sin(th)/(1+3*pow(cos(th),2));
  vel[0] += M*chi*s_xi;
  vel[1] += M*chi/xi*s_th;
  if constexpr (dynamics == Dynamics::zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., -M*chi*(9*pow(sin(th),2)-4)*cos(th)/(3*pow(cos(th),2)+1));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p3o2)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 1/xi};
  if constexpr (dynamics == Dynamics::Pauli)
    return vel;
  auto s_xi = cos(th);
  auto s_th = -sin(th);
  vel[0] += M*chi*s_xi;
  vel[1] += M*chi/xi*s_th;
  if constexpr (dynamics == Dynamics::zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p3o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*cos(th));
}

} // Pauli_Hydrogen
