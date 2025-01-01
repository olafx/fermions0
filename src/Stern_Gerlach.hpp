#pragma once
#include <span>
#include <complex>
#include <vector>
#include "random.hpp"

namespace Stern_Gerlach
{

enum Dynamics
{ Bohmian, Pauli, zigzag
};
using namespace std::complex_literals;

template <Dynamics dynamics>
void gen_ic
( std::span<std::vector<double> *> paths, pcg_xsl_rr_128_64::State &rng,
  std::array<double, 3> sig
)
{ for (auto path : paths)
  { path->insert(path->end(),
      {0, rng.normal()*sig[0],
          rng.normal()*sig[1],
          rng.normal()*sig[2]});
    if constexpr (dynamics == zigzag)
      path->push_back(rng.uniform() < .5 ? -1. : +1.);
  }
}

double dlnrhox
( double t, double x, double sig2x, double px
)
{ return (px*t-x)/(sig2x+pow(t,2)/(4*sig2x));
}

double dlnrhoy
( double t, double y, double sig2y
)
{ return -y/(sig2y+pow(t,2)/(4*sig2y));
}

std::array<double, 2> dlnrhozpm
( double t, double z, double sig2z, double b, double ti, double tf
)
{ auto den = sig2z+pow(t,2)/(4*sig2z);
  if (t < ti)
    return
    { -z/den,
      -z/den
    };
  else if (t < tf)
    return
    { -(z-.5*b*pow(t-ti,2))/den,
      -(z+.5*b*pow(t-ti,2))/den
    };
  else
    return
    { -(z-.5*b*pow(tf-ti,2)-b*(tf-ti)*(t-tf))/den,
      -(z+.5*b*pow(tf-ti,2)+b*(tf-ti)*(t-tf))/den
    };
}

std::complex<double> dlnpsizppsizm
( double t, double z, double sig2z, double b, double ti, double tf
)
{ auto den1 = sig2z+pow(t,2)/(4*sig2z);
  auto den2 = 4*pow(sig2z,2)+pow(t,2);
  if (t < ti)
    return -z/den1;
  else if (t < tf)
    return (-4*z*sig2z+1i*b*pow(t-ti,2)*t)/den2-2i*b*(t-ti);
  else
    return (-4*z*sig2z+1i*b*pow(tf-ti,2)*t+2i*b*(tf-ti)*(t-tf)*t)/den2-2i*b*(tf-ti);
}

// without prefactor
std::array<double, 2> rhozpm
( double t, double z, double sig2z, double b, double ti, double tf
)
{ auto den = 2*sig2z+pow(t,2)/(2*sig2z);
  if (t < ti)
    return
    { exp(-pow(z,2)/den),
      exp(-pow(z,2)/den)
    };
  else if (t < tf)
    return
    { exp(-pow(z-.5*b*pow(t-ti,2),2)/den),
      exp(-pow(z+.5*b*pow(t-ti,2),2)/den)
    };
  else
    return
    { exp(-pow(z-.5*b*pow(tf-ti,2)-b*(tf-ti)*(t-tf),2)/den),
      exp(-pow(z+.5*b*pow(tf-ti,2)+b*(tf-ti)*(t-tf),2)/den)
    };
}

// without prefactor
std::complex<double> psizppsizm
( double t, double z, double sig2z, double b, double ti, double tf
)
{ if (t < ti)
    return exp(-pow(z,2)/(2*sig2z+pow(t,2)/(2*sig2z)));
  else if (t < tf)
    return exp(
      -pow(z-.5*b*pow(t-ti,2),2)/(4*sig2z-2i*t)
      -pow(z+.5*b*pow(t-ti,2),2)/(4*sig2z+2i*t)
      -2i*b*(t-ti)*z);
  else
    return exp(
      -pow(z-.5*b*pow(tf-ti,2)-b*(tf-ti)*(t-tf),2)/(4*sig2z-2i*t)
      -pow(z+.5*b*pow(tf-ti,2)+b*(tf-ti)*(t-tf),2)/(4*sig2z+2i*t)
      -2i*b*(tf-ti)*z);
}

template <Dynamics dynamics>
std::array<double, 3> vel
( double t, std::array<double, 3> X, std::array<double, 3> sig, double px, double b, double ti, double tf,
  std::array<std::complex<double>, 2> c, int chi = 0
)
{ std::array sig2
  { pow(sig[0], 2),
    pow(sig[1], 2),
    pow(sig[2], 2)
  };
  std::array c2
  { std::norm(c[0]),
    std::norm(c[1])
  };
  auto c2_ = std::conj(c[0])*c[1];

  std::array<double, 3> vel_;

  auto rhozpm_ = rhozpm(t, X[2], sig2[2], b, ti, tf);

  vel_[0] = t*(X[0]-px*t)/(4*pow(sig2[0],2)+pow(t,2))+px;
  vel_[1] = t*X[1]/(4*pow(sig2[1],2)+pow(t,2));
  if (t < ti)
    vel_[2] = t*X[2]/(4*pow(sig2[2],2)+pow(t,2));
  else if (t < tf)
  { std::array a
    { t*(X[2]-.5*b*pow(t-ti,2))/(4*pow(sig2[2],2)+pow(t,2))+b*(t-ti),
      t*(X[2]+.5*b*pow(t-ti,2))/(4*pow(sig2[2],2)+pow(t,2))-b*(t-ti)
    };
    vel_[2] = (c2[0]*rhozpm_[0]*a[0]+c2[1]*rhozpm_[1]*a[1])/(c2[0]*rhozpm_[0]+c2[1]*rhozpm_[1]);
  }
  else
  { std::array a
    { t*(X[2]-.5*b*pow(tf-ti,2)-b*(tf-ti)*(t-tf))/(4*pow(sig2[2],2)+pow(t,2))+b*(tf-ti),
      t*(X[2]+.5*b*pow(tf-ti,2)+b*(tf-ti)*(t-tf))/(4*pow(sig2[2],2)+pow(t,2))-b*(tf-ti)
    };
    vel_[2] = (c2[0]*rhozpm_[0]*a[0]+c2[1]*rhozpm_[1]*a[1])/(c2[0]*rhozpm_[0]+c2[1]*rhozpm_[1]);
  }
  if constexpr (dynamics == Bohmian)
    return vel_;

  auto dlnrhox_ = dlnrhox(t, X[0], sig2[0], px);
  auto dlnrhoy_ = dlnrhoy(t, X[1], sig2[1]);
  auto dlnpsizppsizm_ = dlnpsizppsizm(t, X[2], sig2[2], b, ti, tf);
  auto psizppsizm_ = psizppsizm(t, X[2], sig2[2], b, ti, tf);

  auto den = c2[0]*rhozpm_[0]+c2[1]*rhozpm_[1];
  auto Pauli_yz = (c2[0]*rhozpm_[0]-c2[1]*rhozpm_[1])/den*dlnrhoy_;
  auto Pauli_xz = (c2[0]*rhozpm_[0]-c2[1]*rhozpm_[1])/den*dlnrhox_;
  auto Pauli_zy = 2*std::imag(c2_*psizppsizm_*dlnpsizppsizm_)/den;
  auto Pauli_zx = 2*std::real(c2_*psizppsizm_*dlnpsizppsizm_)/den;
  auto Pauli_xy = 2*std::imag(c2_*psizppsizm_)/den*dlnrhox_;
  auto Pauli_yx = 2*std::real(c2_*psizppsizm_)/den*dlnrhoy_;
  vel_[0] += .5*(Pauli_yz-Pauli_zy);
  vel_[1] += .5*(Pauli_zx-Pauli_xz);
  vel_[2] += .5*(Pauli_xy-Pauli_yx);
  if constexpr (dynamics == Pauli)
    return vel_;

  vel_[0] += 2*std::real(c2_*psizppsizm_)/den*chi;
  vel_[1] += 2*std::imag(c2_*psizppsizm_)/den*chi;
  vel_[2] += (c2[0]*rhozpm_[0]-c2[1]*rhozpm_[1])/den*chi;
  if constexpr (dynamics == zigzag)
    return vel_;
}

template <Dynamics dynamics>
requires (dynamics == zigzag)
double rate
( double t, std::array<double, 3> X, std::array<double, 3> sig, double px, double b, double ti, double tf,
  std::array<std::complex<double>, 2> c, int chi = 0
)
{ std::array sig2
  { pow(sig[0], 2),
    pow(sig[1], 2),
    pow(sig[2], 2)
  };
  std::array c2
  { std::norm(c[0]),
    std::norm(c[1])
  };
  auto c2_ = std::conj(c[0])*c[1];

  auto dlnrhox_ = dlnrhox(t, X[0], sig2[0], px);
  auto dlnrhoy_ = dlnrhoy(t, X[1], sig2[1]);
  auto dlnrhozpm_ = dlnrhozpm(t, X[2], sig2[2], b, ti, tf);
  auto rhozpm_ = rhozpm(t, X[2], sig2[2], b, ti, tf);
  auto psizppsizm_ = psizppsizm(t, X[2], sig2[2], b, ti, tf);

  auto den = c2[0]*rhozpm_[0]+c2[1]*rhozpm_[1];
  auto tau =
    (-2*std::real(c2_*psizppsizm_)*dlnrhox_
     -2*std::imag(c2_*psizppsizm_)*dlnrhoy_
     -(c2[0]*rhozpm_[0]*dlnrhozpm_[0]-c2[1]*rhozpm_[1]*dlnrhozpm_[1]))/den;
  return std::max(0., tau*chi);
}

} // Stern_Gerlach
