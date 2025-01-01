#pragma once
#include <array>
#include <cmath>

namespace math_util
{

enum CoordinateSystem
{ Cartesian, spherical
};

// spherical to Cartesian
std::array<double, 3> s_to_C
( std::array<double, 3> Xs
)
{ return
  { Xs[0]*sin(Xs[1])*cos(Xs[2]),
    Xs[0]*sin(Xs[1])*sin(Xs[2]),
    Xs[0]*cos(Xs[1])
  };
}

// Cartesian to spherical
std::array<double, 3> C_to_s
( std::array<double, 3> Xc
)
{ double r = sqrt(pow(Xc[0],2)+pow(Xc[1],2)+pow(Xc[2],2));
  return
  { r,
    acos(Xc[2]/r),
    atan2(Xc[1], Xc[0])
  };
}

} // math_util
