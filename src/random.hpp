#pragma once
#include <cstdint>
#include <cmath>
#include <numbers>
#include <array>

// PCG XSL-RR generator, 64 bit output, 128 bit state. Adapted from the PCG
// library. Copyright M.E. O'Neill (2014), Apache License 2.0.
namespace pcg_xsl_rr_128_64
{

using namespace std::numbers;

struct State
{
  static constexpr auto mult = (static_cast<__uint128_t>(2549297995355413924)<<64)+4865540595714422341;
  __uint128_t state, inc;
  double normal_spare;
  bool normal_has_spare = false;

  State
  ( __uint128_t state_seed, __uint128_t inc_seed
  )
  { state = 0;
    inc = inc_seed<<1|1; // inc must be odd
    state = state*mult+inc;
    state += state_seed;
    state = state*mult+inc;
  }

  uint64_t next
  ()
  { state = state*mult+inc;
    uint64_t x = static_cast<uint64_t>(state>>64)^state;
    size_t rot = state>>122;
    return x>>rot|x<<(-rot&63);
  }

  double uniform
  ()
  { constexpr auto r_max_uint64_p1 = 1./(__uint128_t {1}<<64);
    return next()*r_max_uint64_p1;
  }

  // returns in spherical coordinates (physics norm): (r,theta,phi)
  std::array<double, 3> uniform_unit_ball
  ()
  { return
    { cbrt(uniform()),
      acos(2*uniform()-1),
      2*pi*uniform()
    };
  }

  // Marsaglia polar method
  double normal
  ()
  { double u1, u2, s;
    if (normal_has_spare)
    { normal_has_spare = false;
      return normal_spare;
    }
    for (;;)
    { u1 = uniform()*2-1;
      u2 = uniform()*2-1;
      s = pow(u1,2)+pow(u2,2);
      if (s != 0 && s < 1)
        break;
    }
    [[assume(s < 2)]];
    s = sqrt(-2*log(s)/s);
    normal_spare = u1*s;
    normal_has_spare = true;
    return u2*s;
  }

  double Cauchy
  ()
  { return tan(pi*(uniform()-.5));
  }
};

} // pcg_xsl_rr_128_64
