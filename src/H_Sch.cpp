#include "H_Sch.hpp"
#include "integrate.hpp"
#include "py_util.hpp"

constexpr __uint128_t inc_seed = 2036-8-12;
namespace S_H = Schrodinger_Hydrogen;

template <S_H::Orbital orbital, S_H::Dynamics dynamics, typename... A>
static PyObject *trajectories(A... a) {}

template <S_H::Orbital orbital>
static PyObject *sample
( double a, size_t n, uint64_t seed, bool chi_also, PyObject *c = nullptr
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  std::array<std::complex<double>, 2> c_ {};
  if constexpr (orbital == S_H::_100_210)
    c_ = py_util::from_complex_list<2>(c);
  PyGILState_Release(gil);
  auto *samples = new std::vector<double>((chi_also ? 4 : 3)*n);
  auto n_threads = omp_get_max_threads();
  std::vector<pcg_xsl_rr_128_64::State> rngs;
  for (size_t i = 0; i < n_threads; i++)
    rngs.push_back(pcg_xsl_rr_128_64::State {seed+i, inc_seed+i});
  auto rho_0 = [&](std::array<double, 3> Xs)
  { return S_H::rho_0<orbital>(Xs, c_);
  };
  size_t attempts;
  if (!chi_also)
    attempts = S_H::gen_ic<S_H::Pauli>(*samples, rngs, a, rho_0);
  else
    attempts = S_H::gen_ic<S_H::zigzag>(*samples, rngs, a, rho_0);

  gil = PyGILState_Ensure();
  auto *np_array = py_util::to_np_array<NPY_DOUBLE>(samples, chi_also ? 4 : 3);
  auto *tuple = py_util::to_tuple(attempts, np_array);
  PyGILState_Release(gil);
  return tuple;
}

// analytical Pauli
template <S_H::Orbital orbital, S_H::Dynamics dynamics>
requires (dynamics == S_H::Pauli)
static PyObject *trajectories
( double a, double om1, PyObject *X0, double tf, size_t n, bool print_progress
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto [X0_dims, X0_data] = py_util::from_np_array<NPY_DOUBLE, double>(X0);
  PyGILState_Release(gil);
  auto tauf = om1*tf;
  std::vector<std::vector<double> *> paths(X0_dims[0]); // spherical
  for (size_t i = 0; i < paths.size(); i++)
  { paths[i] = new std::vector<double>(4*n);
    auto &path = *paths[i];
    auto X0s = math_util::C_to_s({X0_data[3*i], X0_data[3*i+1], X0_data[3*i+2]});
    path[0] = 0;
    path[1] = X0s[0];
    path[2] = X0s[1];
    path[3] = X0s[2];
    S_H::undim<math_util::spherical, dynamics>(path, a, 1/om1);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < paths.size(); i++)
  { auto &path = *paths[i];
    auto xi0 = path[1], th0 = path[2], phi0 = path[3];
    auto vel_phi = S_H::vel<orbital, dynamics>(0, {xi0, th0, phi0})[2];
    for (size_t j = 1; j < n; j++)
    { auto tau = tauf/(n-1)*j, phi = vel_phi*tau+phi0;
      path[4*j  ] = tau;
      path[4*j+1] = xi0;
      path[4*j+2] = th0;
      path[4*j+3] = phi;
    }
    S_H::s_to_C<S_H::Pauli>(path);
    S_H::redim<math_util::Cartesian, dynamics>(path, a, 1/om1);
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<4, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

// numerical Pauli
template <S_H::Orbital orbital, S_H::Dynamics dynamics>
requires (dynamics == S_H::Pauli)
static PyObject *trajectories
( double a, PyObject *X0, double tf, double abs_tol, size_t max_iter, bool print_progress, double om1 = 0,
  double om12 = 0, double Om = 0, double nu = 0, PyObject *c = nullptr, double sig = 0
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  std::array<std::complex<double>, 2> c_ {};
  if constexpr (orbital == S_H::_100_210)
    c_ = py_util::from_complex_list<2>(c);
  auto [X0_dims, X0_data] = py_util::from_np_array<NPY_DOUBLE, double>(X0);
  PyGILState_Release(gil);
  double tauf = 0, time_scale = 0; // why are these set to 0? would be clearer if they weren't set
  if constexpr (orbital == S_H::_2p_x)
  { tauf = om1*tf;
    time_scale = 1/om1;
  }
  if constexpr (
    orbital == S_H::_100_210 ||
    orbital == S_H::_100_210_sigmoid ||
    orbital == S_H::_100_210_Rabi)
  { tauf = om12*tf;
    time_scale = 1/om12;
  }
  std::vector<std::vector<double> *> paths(X0_dims[0]); // spherical
  for (size_t i = 0; i < paths.size(); i++)
  { auto X0s = math_util::C_to_s({X0_data[3*i], X0_data[3*i+1], X0_data[3*i+2]});
    paths[i] = new std::vector<double> {0, X0s[0], X0s[1], X0s[2]};
    S_H::undim<math_util::spherical, dynamics>(*paths[i], a, time_scale);
  }

  auto vel = [&](double tau, std::array<double, 3> Xs)
  { if constexpr (orbital == S_H::_2p_x)
      return S_H::vel<orbital, dynamics>(tau, Xs);
    if constexpr (orbital == S_H::_100_210)
      return S_H::vel<orbital, dynamics>(tau, Xs, c_);
    if constexpr (orbital == S_H::_100_210_sigmoid)
      return S_H::vel<orbital, dynamics>(tau, Xs, om12/sig);
    if constexpr (orbital == S_H::_100_210_Rabi)
      return S_H::vel<orbital, dynamics>(tau, Xs, om12, Om, nu);
  };
  auto stop = [&](double tau, std::array<double, 3> Xs)
  { return tau >= tauf;
  };

  #pragma omp parallel for
  for (size_t i = 0; i < paths.size(); i++)
  { auto &path = *paths[i];
    size_t iter = Cash_Karp_45::solve<3>(path, vel, stop, abs_tol, max_iter);
    S_H::s_to_C<dynamics>(path);
    S_H::redim<math_util::Cartesian, dynamics>(path, a, time_scale);
    if (print_progress)
    { int n_length = std::to_string(paths.size()).length();
      printf("%*zu/%zu: %zu\n", n_length, i+1, paths.size(), iter);
    }
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<4, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

// numerical zigzag
template <S_H::Orbital orbital, S_H::Dynamics dynamics>
requires (dynamics == S_H::zigzag)
static PyObject *trajectories
( double a, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress, double om1 = 0, double om12 = 0, PyObject *c = nullptr
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  std::array<std::complex<double>, 2> c_ {};
  if constexpr (orbital == S_H::_100_210)
    c_ = py_util::from_complex_list<2>(c);
  auto [X0_dims, X0_data] = py_util::from_np_array<NPY_DOUBLE, double>(X0);
  PyGILState_Release(gil);
  double tauf = 0, timescale = 0;
  if constexpr (orbital == S_H::_100_210)
  { tauf = om12*tf;
    timescale = 1/om12;
  }
  else
  { tauf = om1*tf;
    timescale = 1/om1;
  }
  std::vector<std::vector<double> *> paths(X0_dims[0]); // spherical
  for (size_t i = 0; i < paths.size(); i++)
  { paths[i] = new std::vector<double> {0, X0_data[4*i], X0_data[4*i+1], X0_data[4*i+2], X0_data[4*i+3]};
    S_H::C_to_s<dynamics>(*paths[i]);
    S_H::undim<math_util::spherical, dynamics>(*paths[i], a, timescale);
  }

  auto vel = [&](double tau, std::array<double, 3> Xs, double chi)
  { if constexpr (orbital == S_H::_100_210)
      return S_H::vel<orbital, dynamics>(tau, Xs, c_, M, chi);
    else
      return S_H::vel<orbital, dynamics>(tau, Xs, M, chi);
  };
  auto rate = [&](double tau, std::array<double, 3> Xs, double chi)
  { if constexpr (orbital == S_H::_100_210)
      return S_H::rate<orbital, dynamics>(tau, Xs, c_, M, chi);
    else
      return S_H::rate<orbital, dynamics>(tau, Xs, M, chi);
  };
  auto stop = [&](double tau, std::array<double, 3> Xs, double chi)
  { return tau >= tauf;
  };

  auto n_threads = omp_get_max_threads();
  std::vector<pcg_xsl_rr_128_64::State> rngs;
  for (size_t i = 0; i < n_threads; i++)
    rngs.push_back(pcg_xsl_rr_128_64::State {seed+i, inc_seed+i});
  #pragma omp parallel for
  for (size_t i = 0; i < paths.size(); i++)
  { auto &path = *paths[i];
    size_t iter = Cash_Karp_45::solve<3>(path, rngs[omp_get_thread_num()], vel, rate, stop, abs_tol, p_tol, max_iter);
    S_H::s_to_C<dynamics>(path);
    S_H::redim<math_util::Cartesian, dynamics>(path, a, timescale);
    if (print_progress)
    { int n_length = std::to_string(paths.size()).length();
      printf("%*zu/%zu: %zu\n", n_length, i+1, paths.size(), iter);
    }
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<5, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

extern "C"
{

PyObject *sample_100
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_100>
    (a, n, seed, chi_also);
}

PyObject *sample_200
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_200>
    (a, n, seed, chi_also);
}

PyObject *sample_210
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_210>
    (a, n, seed, chi_also);
}

PyObject *sample_211
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_211>
    (a, n, seed, chi_also);
}

PyObject *sample_2p_x
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_2p_x>
    (a, n, seed, chi_also);
}

PyObject *sample_100_210
( double a, PyObject *c, size_t n, uint64_t seed, bool chi_also
)
{ return sample<S_H::_100_210>
    (a, n, seed, chi_also, c);
}

PyObject *Pauli_100
( double a, double om1, PyObject *X0, double tf, size_t n, bool print_progress
)
{ return trajectories<S_H::_100, S_H::Pauli>
    (a, om1, X0, tf, n, print_progress);
}

PyObject *Pauli_200
( double a, double om1, PyObject *X0, double tf, size_t n, bool print_progress
)
{ return trajectories<S_H::_200, S_H::Pauli>
    (a, om1, X0, tf, n, print_progress);
}

PyObject *Pauli_210
( double a, double om1, PyObject *X0, double tf, size_t n, bool print_progress
)
{ return trajectories<S_H::_210, S_H::Pauli>
    (a, om1, X0, tf, n, print_progress);
}

PyObject *Pauli_211
( double a, double om1, PyObject *X0, double tf, size_t n, bool print_progress
)
{ return trajectories<S_H::_211, S_H::Pauli>
    (a, om1, X0, tf, n, print_progress);
}

PyObject *Pauli_2p_x
( double a, double om1, PyObject *X0, double tf, double abs_tol, size_t max_iter, bool print_progress
)
{ return trajectories<S_H::_2p_x, S_H::Pauli>
    (a, X0, tf, abs_tol, max_iter, print_progress, om1);
}

PyObject *Pauli_100_210
( double a, double om12, PyObject *c, PyObject *X0, double tf, double abs_tol, size_t max_iter, bool print_progress
)
{ return trajectories<S_H::_100_210, S_H::Pauli>
    (a, X0, tf, abs_tol, max_iter, print_progress, 0., om12, 0., 0., c);
}

PyObject *Pauli_100_210_sigmoid
( double a, double om12, double sig, PyObject *X0, double tf, double abs_tol, size_t max_iter, bool print_progress
)
{ return trajectories<S_H::_100_210_sigmoid, S_H::Pauli>
    (a, X0, tf, abs_tol, max_iter, print_progress, 0., om12, 0., 0., static_cast<PyObject *>(nullptr), sig);
}

PyObject *Pauli_100_210_Rabi
( double a, double om0, double Om, double nu, PyObject *X0, double tf, double abs_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_100_210_Rabi, S_H::Pauli>
    (a, X0, tf, abs_tol, max_iter, print_progress, 0., om0, Om, nu);
}

PyObject *zigzag_100
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_100, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_200
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_200, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_210
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_210, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_211
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_211, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_2p_x
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<S_H::_2p_x, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_100_210
( double a, double om12, double M, PyObject *c, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol,
  size_t max_iter, bool print_progress
)
{ return trajectories<S_H::_100_210, S_H::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, 0., om12, c);
}

}
