#pragma once
#include <span>
#include <complex>
#include "random.hpp"

namespace double_slit
{

enum Dynamics
{ Bohmian, Pauli, zigzag
};
using namespace std::complex_literals;

template <Dynamics dynamics>
void gen_ic
( std::span<std::vector<double> *> paths, pcg_xsl_rr_128_64::State &rng, std::span<double, 3> sig, double d
)
{ for (size_t i = 0; i < paths.size(); i++)
  { paths[i]->insert(paths[i]->end(),
      {0, rng.normal()*sig[0],
          rng.normal()*sig[1]+(rng.uniform() < .5 ? -d : +d),
          rng.normal()*sig[2]});
    if constexpr (dynamics == zigzag)
      paths[i]->push_back(rng.uniform() < .5 ? -1. : +1.);
  }
}

std::array<std::complex<double>, 3> nab_log_psi
( double t, std::array<double, 3> X, std::array<double, 3> sig, double vx, double d, double m
)
{ std::array sig2t
  { pow(sig[0], 2)+.5i*t/m,
    pow(sig[1], 2)+.5i*t/m,
    pow(sig[2], 2)+.5i*t/m
  };
  std::array psiz // y+d, y-d
  { exp(-pow(X[1]+d,2)/(4.*sig2t[1])),
    exp(-pow(X[1]-d,2)/(4.*sig2t[1]))
  };
  return
  { -(X[0]-vx*t)/(2.*sig2t[0])+1i*m*vx,
    (psiz[0]*(-X[1]-d)+psiz[1]*(-X[1]+d))/(2.*sig2t[1])
      /(psiz[0]+psiz[1]),
    -X[2]/(2.*sig2t[2])
  };
}

template <Dynamics dynamics>
std::array<double, 3> vel
( double t, std::array<double, 3> X, std::array<double, 3> sig, double vx, double d, double m,
  std::array<double, 3> s = {}, int chi = 0
)
{ auto nab_log_psi_ = nab_log_psi(t, X, sig, vx, d, m);
  // only dividing by m at the end
  std::array vel_
  { std::imag(nab_log_psi_[0]),
    std::imag(nab_log_psi_[1]),
    std::imag(nab_log_psi_[2])
  };
  if constexpr (dynamics == Bohmian)
    return
    { vel_[0]/m,
      vel_[1]/m,
      vel_[2]/m
    };
  vel_[0] += std::real(nab_log_psi_[1])*s[2]-std::real(nab_log_psi_[2])*s[1];
  vel_[1] += std::real(nab_log_psi_[2])*s[0]-std::real(nab_log_psi_[0])*s[2];
  vel_[2] += std::real(nab_log_psi_[0])*s[1]-std::real(nab_log_psi_[1])*s[0];
  if constexpr (dynamics == Pauli)
    return
    { vel_[0]/m,
      vel_[1]/m,
      vel_[2]/m
    };
  if constexpr (dynamics == zigzag)
    return
    { vel_[0]/m+chi*s[0],
      vel_[1]/m+chi*s[1],
      vel_[2]/m+chi*s[2]
    };
}

template <Dynamics dynamics>
double rate
( double t, std::array<double, 3> X, std::array<double, 3> sig, double vx, double d, double m,
  std::array<double, 3> s = {}, int chi = 0
)
{ if constexpr (dynamics == zigzag)
  { auto nab_log_psi_ = nab_log_psi(t, X, sig, vx, d, m);
    auto tau = -2*(s[0]*std::real(nab_log_psi_[0])+
                   s[1]*std::real(nab_log_psi_[1])+
                   s[2]*std::real(nab_log_psi_[2]));
    return std::max(0., tau*chi);
  }
}

} // double_slit
