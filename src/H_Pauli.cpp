#include "H_Pauli.hpp"
#include "integrate.hpp"
#include "py_util.hpp"

constexpr __uint128_t inc_seed = 2036-8-12;
namespace P_H = Pauli_Hydrogen;

template <P_H::Orbital orbital, P_H::Dynamics dynamics, typename... A>
static PyObject *trajectories(A... a) {}

template <P_H::Orbital orbital>
static PyObject *sample
( double a, size_t n, uint64_t seed, bool chi_also
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  PyGILState_Release(gil);
  auto *samples = new std::vector<double>((chi_also ? 4 : 3)*n);
  auto n_threads = omp_get_max_threads();
  std::vector<pcg_xsl_rr_128_64::State> rngs;
  for (size_t i = 0; i < n_threads; i++)
    rngs.push_back(pcg_xsl_rr_128_64::State {seed+i, inc_seed+i});
  auto rho_0 = [&](std::array<double, 3> Xs)
  { return P_H::rho_0<orbital>(Xs);
  };
  size_t attempts;
  if (!chi_also)
    attempts = P_H::gen_ic<P_H::Dynamics::Pauli>(*samples, rngs, a, rho_0);
  else
    attempts = P_H::gen_ic<P_H::Dynamics::zigzag>(*samples, rngs, a, rho_0);

  gil = PyGILState_Ensure();
  auto *np_array = py_util::to_np_array<NPY_DOUBLE>(samples, chi_also ? 4 : 3);
  auto *tuple = py_util::to_tuple(attempts, np_array);
  PyGILState_Release(gil);
  return tuple;
}

template <P_H::Orbital orbital, P_H::Dynamics dynamics>
requires (dynamics == P_H::Dynamics::zigzag)
static PyObject *trajectories
( double a, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress, double om1
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto [X0_dims, X0_data] = py_util::from_np_array<NPY_DOUBLE, double>(X0);
  PyGILState_Release(gil);
  double tauf = 0, timescale = 0;
  tauf = om1*tf;
  timescale = 1/om1;
  std::vector<std::vector<double> *> paths(X0_dims[0]); // spherical
  for (size_t i = 0; i < paths.size(); i++)
  { paths[i] = new std::vector<double> {0, X0_data[4*i], X0_data[4*i+1], X0_data[4*i+2], X0_data[4*i+3]};
    P_H::C_to_s<dynamics>(*paths[i]);
    P_H::undim<math_util::spherical, dynamics>(*paths[i], a, timescale);
  }

  auto vel = [&](double tau, std::array<double, 3> Xs, double chi)
  { return P_H::vel<orbital, dynamics>(tau, Xs, M, chi);
  };
  auto rate = [&](double tau, std::array<double, 3> Xs, double chi)
  { return P_H::rate<orbital, dynamics>(tau, Xs, M, chi);
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
    P_H::s_to_C<dynamics>(path);
    P_H::redim<math_util::Cartesian, dynamics>(path, a, timescale);
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

PyObject *sample_1_0_1o2_p1o2
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<P_H::_1_0_1o2_p1o2>
    (a, n, seed, chi_also);
}

PyObject *sample_2_0_1o2_p1o2
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<P_H::_2_0_1o2_p1o2>
    (a, n, seed, chi_also);
}

PyObject *sample_2_1_1o2_p1o2
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<P_H::_2_1_1o2_p1o2>
    (a, n, seed, chi_also);
}

PyObject *sample_2_1_3o2_p1o2
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<P_H::_2_1_3o2_p1o2>
    (a, n, seed, chi_also);
}

PyObject *sample_2_1_3o2_p3o2
( double a, size_t n, uint64_t seed, bool chi_also
)
{ return sample<P_H::_2_1_3o2_p3o2>
    (a, n, seed, chi_also);
}

PyObject *zigzag_1_0_1o2_p1o2
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<P_H::_1_0_1o2_p1o2, P_H::Dynamics::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_2_0_1o2_p1o2
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<P_H::_2_0_1o2_p1o2, P_H::Dynamics::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_2_1_1o2_p1o2
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<P_H::_2_1_1o2_p1o2, P_H::Dynamics::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_2_1_3o2_p1o2
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<P_H::_2_1_3o2_p1o2, P_H::Dynamics::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

PyObject *zigzag_2_1_3o2_p3o2
( double a, double om1, double M, PyObject *X0, double tf, uint64_t seed, double abs_tol, double p_tol, size_t max_iter,
  bool print_progress
)
{ return trajectories<P_H::_2_1_3o2_p3o2, P_H::Dynamics::zigzag>
    (a, M, X0, tf, seed, abs_tol, p_tol, max_iter, print_progress, om1);
}

}
