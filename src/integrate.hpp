#pragma once
#include <cmath>
#include <array>
#include <span>
#include <vector>
#include <functional>
#include "random.hpp"

// Cash-Karp integrator, adaptive 5th order Runge-Kutta with 4th order error
// estimates.
// J. R. Cash, A. H. Karp. "A variable order Runge-Kutta method for initial
// value problems with rapidly varying right-hand sides"
namespace Cash_Karp_45
{

// Butcher tableau
constexpr double
  c1 = 0, c2 = 1./5, c3 = 3./10, c4 = 3./5, c5 = 1, c6 = 7./8,
  b51 = 37./378, b52 = 0, b53 = 250./621, b54 = 125./594, b55 = 0, b56 = 512./1771,
  b41 = 2825./27648, b42 = 0, b43 = 18575./48384, b44 = 13525./55296, b45 = 277./14336, b46 = 1./4,
  a21 = 1./5,
  a31 = 3./40, a32 = 9./40,
  a41 = 3./10, a42 = -9./10, a43 = 6./5,
  a51 = -11./54, a52 = 5./2, a53 = -70./27, a54 = 35./27,
  a61 = 1631./55296, a62 = 175./512, a63 = 575./13824, a64 = 44275./110592, a65 = 253./4096;

// x'=F(t,x)
template <size_t n>
std::tuple<double, std::array<double, n>> eval
( auto F, double t, std::array<double, n> x, double dt
)
{ auto dx1 = F(t+c1*dt, x);
  for (size_t i = 0; i < n; i++) dx1[i] *= dt;
  auto x2 = x;
  for (size_t i = 0; i < n; i++) x2[i] += a21*dx1[i];
  auto dx2 = F(t+c2*dt, x2);
  for (size_t i = 0; i < n; i++) dx2[i] *= dt;
  auto x3 = x;
  for (size_t i = 0; i < n; i++) x3[i] += a31*dx1[i]+a32*dx2[i];
  auto dx3 = F(t+c3*dt, x3);
  for (size_t i = 0; i < n; i++) dx3[i] *= dt;
  auto x4 = x;
  for (size_t i = 0; i < n; i++) x4[i] += a41*dx1[i]+a42*dx2[i]+a43*dx3[i];
  auto dx4 = F(t+c4*dt, x4);
  for (size_t i = 0; i < n; i++) dx4[i] *= dt;
  auto x5 = x;
  for (size_t i = 0; i < n; i++) x5[i] += a51*dx1[i]+a52*dx2[i]+a53*dx3[i]+a54*dx4[i];
  auto dx5 = F(t+c5*dt, x5);
  for (size_t i = 0; i < n; i++) dx5[i] *= dt;
  auto x6 = x;
  for (size_t i = 0; i < n; i++) x6[i] += a61*dx1[i]+a62*dx2[i]+a63*dx3[i]+a64*dx4[i]+a65*dx5[i];
  auto dx6 = F(t+c6*dt, x6);
  for (size_t i = 0; i < n; i++) dx6[i] *= dt;

  std::array<double, n> dx_4, dx_5, err_4;
  for (size_t i = 0; i < n; i++) dx_4[i] = b41*dx1[i]+b42*dx2[i]+b43*dx3[i]+b44*dx4[i]+b45*dx5[i]+b46*dx6[i];
  for (size_t i = 0; i < n; i++) dx_5[i] = b51*dx1[i]+b52*dx2[i]+b53*dx3[i]+b54*dx4[i]+b55*dx5[i]+b56*dx6[i];
  for (size_t i = 0; i < n; i++) err_4[i] = dx_5[i]-dx_4[i];
  double err_4_sq = 0;
  for (size_t i = 0; i < n; i++) err_4_sq += err_4[i]*err_4[i];
  return {sqrt(err_4_sq), dx_5};
}

// x' = F(t,x)
// stop if stop(t,x)
template <size_t n>
size_t solve
( std::vector<double> &data, auto F, auto stop, double abs_tol, size_t max_iter = 0, size_t min_iter = 0,
  double min_iter_tf = 0
)
{ double dt = 1, t = data[0];
  std::array<double, n> x;
  for (size_t i = 0; i < n; i++)
    x[i] = data[i+1];
  size_t iter = 0;

  auto save = [&]()
  { data.push_back(t);
    for (size_t i = 0; i < n; i++)
      data.push_back(x[i]);
  };

  for (;;)
  { for (;;)
    { if (max_iter != 0 && iter == max_iter) [[unlikely]]
        goto end;
      auto [err_4, dx] = eval(F, t, x, dt);
      [[assume(abs_tol/err_4 > 0)]];
      auto e_4_r = pow(abs_tol/err_4, .2);
      if (err_4 > abs_tol) [[unlikely]]
      { dt *= .9*std::max(0.1, e_4_r);
        continue; 
      }
      if (min_iter != 0 && iter < min_iter)
      { auto dt_max = min_iter_tf/min_iter;
        if (dt > dt_max)
        { dt = dt_max;
          continue;
        }
      }
      iter++;
      t += dt;
      for (size_t i = 0; i < n; i++)
        x[i] += dx[i];
      dt *= .9*std::min(10., e_4_r);
      break;
    }
    save();
    if (stop(t, x)) [[unlikely]]
      break;
  }
  end:
  data.shrink_to_fit();
  return iter;
}

// x' = F(t,x,chi)
// chi=+-1 with rate R(t,x,chi)
// stop if stop(t,x,chi)
template <size_t n, bool save_only_jumps = false>
size_t solve
( std::vector<double> &data, pcg_xsl_rr_128_64::State &state, auto F, auto R, auto stop, double abs_tol, double p_tol,
  size_t max_iter = 0, size_t min_iter = 0, double min_iter_tf = 0
)
{ auto dt = 1., t = data[0];
  std::array<double, n> x;
  for (size_t i = 0; i < n; i++)
    x[i] = data[i+1];
  auto chi = static_cast<int>(data[n+1]);
  size_t iter = 0;

  auto save = [&]()
  { data.push_back(t);
    for (size_t i = 0; i < n; i++)
      data.push_back(x[i]);
    data.push_back(static_cast<double>(chi));
  };

  for (;;)
  { auto F_ = [&](double t_, std::array<double, n> x_){ return F(t_, x_, chi); };
    for (;;)
    { if (max_iter != 0 && iter == max_iter) [[unlikely]]
        goto end;
      auto r1 = R(t, x, chi);
      auto [err_4, dx] = eval(F_, t, x, dt);
      [[assume(abs_tol/err_4 > 0)]];
      auto e_4_r = pow(abs_tol/err_4, .2);
      if (err_4 > abs_tol) [[unlikely]]
      { dt *= .9*std::max(.1, e_4_r);
        continue;
      }
      if (min_iter != 0 && iter < min_iter)
      { auto dt_max = min_iter_tf/min_iter;
        if (dt > dt_max)
        { dt = dt_max;
          continue;
        }
      }
      auto x_try = x;
      for (size_t i = 0; i < n; i++)
        x_try[i] += dx[i];
      auto r2 = R(t+dt, x_try, chi);
      auto r = .5*(r1+r2);
      if (r*dt > p_tol)
      { dt = p_tol/r;
        continue;
      }
      iter++;
      t += dt;
      x = x_try;
      if (state.uniform() < r*dt) [[unlikely]]
      { chi = -chi;
        if constexpr (save_only_jumps)
          save();
      }
      if constexpr (!save_only_jumps)
        save();
      dt *= .9*std::min(10., e_4_r);
      if (r*dt > p_tol)
        dt = p_tol/r;
      break;
    }
    if (stop(t, x, chi)) [[unlikely]]
      break;
  }
  end:
  data.shrink_to_fit();
  return iter;
}

} // Cash_Karp_45
