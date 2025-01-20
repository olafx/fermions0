#pragma once
#include <omp.h>
#include <thread>
#include <complex>
#include <vector>
#include <span>
#include "H_Sch.hpp"
#include "math_util.hpp"
#include "random.hpp"

namespace Dirac_Hydrogen
{

enum Orbital
{ _1_0_1o2_p1o2, _2_1_3o2_p1o2, _2_1_3o2_p3o2
};
using namespace std::numbers;
using namespace std::complex_literals;
namespace S_H = Schrodinger_Hydrogen;
using S_H::Dynamics, S_H::undim, S_H::redim, S_H::s_to_C, S_H::C_to_s, S_H::gen_ic;

template <Orbital orbital>
double rho_0
( std::array<double, 3> Xs, double alpha
)
{ auto [xi, th, phi] = Xs;
  if constexpr (orbital == _1_0_1o2_p1o2)
  { auto Jp = sqrt(1-pow(alpha,2));
    auto rho = 2*xi;
    return 2*pow(rho,2*Jp-2)*exp(-rho)*inv_pi/tgamma(2*Jp+1);
  }
  if constexpr (orbital == _2_1_3o2_p1o2)
  { auto Jp = sqrt(4-pow(alpha,2));
    auto rho = xi;
    return pow(rho,2*Jp-2)*exp(-rho)*(3*pow(cos(th),2)+1)*inv_pi/8/tgamma(2*Jp+1);
  }
  if constexpr (orbital == _2_1_3o2_p3o2)
  { auto Jp = sqrt(4-pow(alpha,2));
    auto rho = xi;
    return 3*pow(rho,2*Jp-2)*exp(-rho)*pow(sin(th),2)*inv_pi/8/tgamma(2*Jp+1);
  } 
}

template <Orbital orbital, Dynamics dynamics, typename ...A>
auto vel(A... a) {}
template <Orbital orbital, Dynamics dynamics, typename ...A>
auto rate(A... a) {}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _1_0_1o2_p1o2 && dynamics == Dynamics::zigzag)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  auto Jp = sqrt(1-pow(alpha,2));
  auto s_5_phi = alpha*sin(th);
  auto s_5_norm = s_5_phi;
  auto s_norm = sqrt(1-pow(s_5_norm,2));
  auto z_s = -(1-Jp)*pow(sin(th),2)+1;
  auto chi_s_pm_xi = chi*s_norm*(sqrt(1-pow(z_s,2))*sin(th)+z_s*cos(th));
  auto chi_s_pm_th = chi*s_norm*(sqrt(1-pow(z_s,2))*cos(th)-z_s*sin(th));
  auto chi_s_pm_ph = s_5_phi;
  return
  { M*chi_s_pm_xi,
    M*chi_s_pm_th/xi,
    M*chi_s_pm_ph/(xi*sin(th))
  };
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _1_0_1o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., chi*pow(M,2)*alpha*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p1o2 && dynamics == Dynamics::zigzag)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  auto Jp = sqrt(4-pow(alpha,2));
  auto s_5_phi = alpha*(9*pow(cos(th),2)-1)*sin(th)/(8-6*pow(sin(th),2));
  auto s_5_norm = abs(s_5_phi);
  auto s_norm = sqrt(1-pow(s_5_norm,2));
  auto z_s = ((9*(2-Jp)*pow(cos(th),2)+10*(Jp-1))*pow(cos(th),2)-Jp)/(8-6*pow(sin(th),2));
  auto chi_s_pm_xi = chi*s_norm*(sqrt(1-pow(z_s,2))*sin(th)+z_s*cos(th));
  auto chi_s_pm_th = chi*s_norm*(sqrt(1-pow(z_s,2))*cos(th)-z_s*sin(th));
  auto chi_s_pm_ph = s_5_phi;
  return
  { M*chi_s_pm_xi,
    M*chi_s_pm_th/xi,
    M*chi_s_pm_ph/(xi*sin(th))
  };
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p1o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., -chi*pow(M,2)*alpha*(9*pow(sin(th),2)-4)*cos(th)/(8-6*pow(sin(th),2)));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p3o2 && dynamics == Dynamics::zigzag)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  auto Jp = sqrt(4-pow(alpha,2));
  auto s_5_phi = alpha/2*sin(th);
  auto s_5_norm = abs(s_5_phi);
  auto s_norm = sqrt(1-pow(s_5_norm,2));
  auto z_s = -(2-Jp)/2*pow(sin(th),2)+1;
  auto chi_s_pm_xi = chi*s_norm*(sqrt(1-pow(z_s,2))*sin(th)+z_s*cos(th));
  auto chi_s_pm_th = chi*s_norm*(sqrt(1-pow(z_s,2))*cos(th)-z_s*sin(th));
  auto chi_s_pm_ph = s_5_phi;
  return
  { M*chi_s_pm_xi,
    M*chi_s_pm_th/xi,
    M*chi_s_pm_ph/(xi*sin(th))
  };
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2_1_3o2_p3o2 && dynamics == Dynamics::zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi, double alpha
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., chi*pow(M,2)*alpha/2*cos(th));
}

} // Dirac_Hydrogen
