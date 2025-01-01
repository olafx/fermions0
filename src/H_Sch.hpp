#pragma once
#include <omp.h>
#include <thread>
#include <complex>
#include <vector>
#include <span>
#include "math_util.hpp"
#include "random.hpp"

namespace Schrodinger_Hydrogen
{

using namespace std::numbers;

enum Dynamics
{ Pauli, zigzag
};
enum Orbital
{ _100, _200, _210, _211, _2p_x, _100_210, _100_210_sigmoid, _100_210_Rabi
};
using namespace std::complex_literals;

// undimensionalize (before integration)
template <math_util::CoordinateSystem coordinate_system, Dynamics dynamics>
void undim
( std::span<double> path, double length_scale, double time_scale
)
{ constexpr size_t spacing = dynamics == Pauli ? 4 : 5;
  for (size_t i = 0; i < path.size()/spacing; i++)
  { if constexpr (coordinate_system == math_util::spherical)
    { path[spacing*i  ] /= time_scale;
      path[spacing*i+1] /= length_scale;
    }
    if constexpr (coordinate_system == math_util::Cartesian)
    { path[spacing*i  ] /= time_scale;
      path[spacing*i+1] /= length_scale;
      path[spacing*i+2] /= length_scale;
      path[spacing*i+3] /= length_scale;
    }
  }
}

// redimensionalize (after integration)
template <math_util::CoordinateSystem coordinate_system, Dynamics dynamics>
void redim
( std::span<double> path, double length_scale, double time_scale
)
{ constexpr size_t spacing = dynamics == Pauli ? 4 : 5;
  for (size_t i = 0; i < path.size()/spacing; i++)
  { if constexpr (coordinate_system == math_util::spherical)
    { path[spacing*i  ] *= time_scale;
      path[spacing*i+1] *= length_scale;
    }
    if constexpr (coordinate_system == math_util::Cartesian)
    { path[spacing*i  ] *= time_scale;
      path[spacing*i+1] *= length_scale;
      path[spacing*i+2] *= length_scale;
      path[spacing*i+3] *= length_scale;
    }
  }
}

// spherical to Cartesian
template <Dynamics dynamics>
void s_to_C
( std::span<double> path
)
{ constexpr size_t spacing = dynamics == Pauli ? 4 : 5;
  for (size_t i = 0; i < path.size()/spacing; i++)
  { auto XC = math_util::s_to_C({path[spacing*i+1], path[spacing*i+2], path[spacing*i+3]});
    path[spacing*i+1] = XC[0];
    path[spacing*i+2] = XC[1];
    path[spacing*i+3] = XC[2];
  }
}

// Cartesian to spherical
template <Dynamics dynamics>
void C_to_s
( std::span<double> path
)
{ constexpr size_t spacing = dynamics == Pauli ? 4 : 5;
  for (size_t i = 0; i < path.size()/spacing; i++)
  { auto Xs = math_util::C_to_s({path[spacing*i+1], path[spacing*i+2], path[spacing*i+3]});
    path[spacing*i+1] = Xs[0];
    path[spacing*i+2] = Xs[1];
    path[spacing*i+3] = Xs[2];
  }
}

template <Orbital orbital>
double rho_0
( std::array<double, 3> Xs, std::array<double, 2> c = {}
)
{ auto [xi, th, phi] = Xs;
  if constexpr (orbital == _100)
    return exp(-2*xi)*inv_pi;
  if constexpr (orbital == _200)
    return exp(-xi)*pow(1-.5*xi,2)*inv_pi/8;
  if constexpr (orbital == _210)
    return exp(-xi)*pow(xi*cos(th),2)*inv_pi/32;
  if constexpr (orbital == _211)
    return exp(-xi)*pow(xi*sin(th),2)*inv_pi/64;
  if constexpr (orbital == _2p_x)
    return exp(-xi)*pow(sin(th),2)*pow(cos(phi),2)*inv_pi/32;
  if constexpr (orbital == _100_210)
  { auto psi_100 = exp(-xi)*inv_sqrtpi;
    auto psi_210 = exp(-.5*xi)*xi*cos(th)/sqrt(32*pi);
    return pow(c[0]*psi_100+c[1]*psi_210,2);
  }
}

template <Dynamics dynamics>
size_t gen_ic
( std::vector<double> &samples, std::vector<pcg_xsl_rr_128_64::State> &rngs, double a, auto rho_0
)
{ constexpr size_t spacing = dynamics == zigzag ? 4 : 3;
  auto n = samples.size()/spacing;
  auto n_threads = rngs.size();
  size_t attempts = 0;
  thread_local double M = 1.;
  #pragma omp parallel num_threads(n_threads) reduction(+:attempts)
  { size_t i_thread = omp_get_thread_num();
    auto i_start = i_thread*(n/n_threads)+std::min(i_thread, n % n_threads);
    auto i_end = i_start+(n/n_threads)+(i_thread < n % n_threads ? 1 : 0);
    auto &rng = rngs[i_thread];
    for (size_t i = i_start; i < i_end; i++)
      for (;;)
      { attempts++;
        std::array XC {rng.Cauchy(),
                       rng.Cauchy(),
                       rng.Cauchy()};
        auto q = 1./((1+pow(XC[0],2))*
                     (1+pow(XC[1],2))*
                     (1+pow(XC[2],2)));
        auto A = rho_0(math_util::C_to_s(XC))/(M*q);
        if (A > 1) [[unlikely]]
        { M *= 1.1;
          continue;
        }
        if (rng.uniform() < A)
        { samples[spacing*i  ] = XC[0]*a;
          samples[spacing*i+1] = XC[1]*a;
          samples[spacing*i+2] = XC[2]*a;
          if constexpr (dynamics == zigzag)
            samples[spacing*i+3] = rng.uniform() < .5 ? -1. : +1.;
          break;
        }
      }
  }
  return attempts;
}

template <Orbital orbital, Dynamics dynamics, typename ...A>
auto vel(A... a) {}
template <Orbital orbital, Dynamics dynamics, typename ...A>
auto rate(A... a) {}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 2/xi};
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += M*chi*cos(th);
  vel[1] += -M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100 && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., 2*M*chi*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _200)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 1/xi*(1/(1-.5*xi)+1)};
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += M*chi*cos(th);
  vel[1] += -M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _200 && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*(1/(1-.5*xi)+1)*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _210)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 1/xi*(1-1/xi)};
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += M*chi*cos(th);
  vel[1] += -M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _210 && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., -M*chi*((2/xi-1)*cos(th)+sin(th)*tan(th)/xi));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _211)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array {0., 0., 1/xi};
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += M*chi*cos(th);
  vel[1] += -M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _211 && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2p_x)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  auto vel = std::array
  { -2/xi*tan(phi),
    -2/pow(xi,2)/tan(th)*tan(phi),
    1/xi*(1-2/(xi*pow(sin(th),2)))
  };
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += M*chi*cos(th);
  vel[1] += -M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _2p_x && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  return std::max(0., M*chi*cos(th));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100_210)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, std::array<double, 2> c, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  std::array c2
  { pow(c[0],2),
    pow(c[1],2)
  };
  auto c_ = c[0]*c[1];
  auto A = c_/sqrt(32)*(1+.5*xi)*exp(-.5*xi)*cos(th)*sin(tau);
  auto B = -c_/sqrt(32)*exp(-.5*xi)*sin(th)*sin(tau);
  auto R = c2[0]*exp(-xi)
    +c2[1]/32*pow(xi,2)*pow(cos(th),2)
    +c_/sqrt(8)*xi*exp(-.5*xi)*cos(th)*cos(tau);
  auto vel = std::array
  { 8./3*A/R,
    8./3*B/(xi*R),
    4./3/(xi*R)*(
      2*c2[0]*exp(-xi)
      +c2[1]/32*pow(xi,2)*pow(cos(th),2)
      +c_/sqrt(8)*1.5*xi*exp(-.5*xi)*cos(th)*cos(tau))
  };
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += 4./3*M*chi*cos(th);
  vel[1] += -4./3*M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100_210 && dynamics == zigzag)
double rate
( double tau, std::array<double, 3> Xs, std::array<double, 2> c, double M, double chi
)
{ auto [xi, th, phi] = Xs;
  std::array c2
  { pow(c[0],2),
    pow(c[1],2)
  };
  auto c_ = c[0]*c[1];
  auto R = c2[0]*exp(-xi)
    +c2[1]/32*pow(xi,2)*pow(cos(th),2)
    +c_/sqrt(8)*xi*exp(-.5*xi)*cos(th)*cos(tau);
  return std::max(0., 8./3*M/R*chi
    *(2*c2[0]*exp(-xi)*cos(th)
      -c2[1]/16*xi*cos(th)*(1-.5*xi*pow(cos(th),2))
      -c_/sqrt(8)*(1-1.5*xi*pow(cos(th),2))*exp(-.5*xi)*cos(tau)));
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100_210_sigmoid)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double om1_sig, double M = 0, double chi = 0
)
{ auto [xi, th, phi] = Xs;
  constexpr auto sigm_scale = 3.;
  auto sigm_x = 2*sigm_scale/(pi*om1_sig)*tau-sigm_scale;
  std::array<double, 2> c;
  c[1] = 1./(1+exp(-sigm_x));
  c[0] = 1-c[1];
  std::array c2
  { pow(c[0],2),
    pow(c[1],2)
  };
  auto c_ = c[0]*c[1];
  auto A = c_/sqrt(32)*(1+.5*xi)*exp(-.5*xi)*cos(th)*sin(tau);
  auto B = -c_/sqrt(32)*exp(-.5*xi)*sin(th)*sin(tau);
  auto R = c2[0]*exp(-xi)
    +c2[1]/32*pow(xi,2)*pow(cos(th),2)
    +c_/sqrt(8)*xi*exp(-.5*xi)*cos(th)*cos(tau);
  auto vel = std::array
  { 8./3*A/R,
    8./3*B/(xi*R),
    4./3/(xi*R)*(
      2*c2[0]*exp(-xi)
      +c2[1]/32*pow(xi,2)*pow(cos(th),2)
      +c_/sqrt(8)*1.5*xi*exp(-.5*xi)*cos(th)*cos(tau))
  };
  if constexpr (dynamics == Pauli)
    return vel;
  vel[0] += 4./3*M*chi*cos(th);
  vel[1] += -4./3*M*chi/xi*sin(th);
  if constexpr (dynamics == zigzag)
    return vel;
}

template <Orbital orbital, Dynamics dynamics>
requires (orbital == _100_210_Rabi)
std::array<double, 3> vel
( double tau, std::array<double, 3> Xs, double om0, double Om, double nu
)
{ auto [xi, th, phi] = Xs;
  auto Om2 = pow(Om,2), nu2 = pow(nu,2), sig = sqrt(Om2+nu2);
  std::array c // a, b
  { (sig+Om)/(2*sig)*exp(.5i*(Om-sig)/om0*tau)+(sig-Om)/(2*sig)*exp(.5i*(Om+sig)/om0*tau),
    nu/(2*sig)*exp(1i*(Om-sig)/(2*om0)*tau)-nu/(2*sig)*exp(1i*(Om+sig)/(2*om0)*tau)
  };
  std::array c2 // a, b
  { std::norm(c[0]),
    std::norm(c[1])
  };
  auto beta_a = 1/sqrt(32), beta_r = xi*beta_a;
  auto xi2 = pow(xi,2), sig2 = pow(sig,2), tau_ = sig/om0*tau;
  auto T = -cos(tau)*sin(tau_)+Om/sig*sin(tau)*(cos(tau_)-1);
  auto Tp = nu/(2*sig)*(Om/sig*cos(tau)*(1-cos(tau_))-sin(tau)*sin(tau_));
  auto D =
    exp(-2*xi)*(sig2+Om2+nu2*cos(tau_))/(2*sig2)
    +pow(beta_r,2)*exp(-xi)*nu2/(2*sig2)*pow(cos(th),2)*(1-cos(tau_))
    +nu/sig*beta_r*exp(-1.5*xi)*cos(th)*(
      Om/sig*cos(tau)*(1-cos(tau_))-sin(tau)*sin(tau_));
  std::array chi // r, theta
  { 1/beta_a*c2[0]*exp(-2*xi)
    +beta_r*c2[1]*pow(cos(th),2)*exp(-xi)*(1-.5*xi)
    +cos(th)*exp(-1.5*xi)*(1-1.5*xi)*Tp,
    -beta_r*c2[1]*exp(-xi)*sin(th)*cos(th)
    -exp(-1.5*xi)*sin(th)*Tp
  };
  auto vel = std::array
  { nu/(3*sqrt2*sig)*(cos(th)*exp(-1.5*xi)*(1+.5*xi))*T/D,
    -nu/(3*sqrt2*sig)*(sin(th)*exp(-1.5*xi)/xi)*T/D,
    nu/(3*sqrt2*sig*xi*D)*(chi[0]+chi[1]/tan(th))
  };
  if constexpr (dynamics == Pauli)
    return vel;
}

} // Schrodinger_Hydrogen
