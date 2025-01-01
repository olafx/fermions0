#include <omp.h>
#include "double_slit.hpp"
#include "integrate.hpp"
#include "py_util.hpp"

constexpr __uint128_t inc_seed = 2036-8-12;

extern "C"
{

PyObject *Bohmian
( PyObject *sig, double d, double vx, double dist, double m, size_t n, uint64_t seed, double abs_tol, size_t max_iter,
  bool print_progress
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto sig_ = py_util::from_float_list<3>(sig);
  PyGILState_Release(gil);
  std::vector<std::vector<double> *> paths(n);
  for (size_t i = 0; i < n; i++)
    paths[i] = new std::vector<double>;
  pcg_xsl_rr_128_64::State rng {seed, inc_seed};
  double_slit::gen_ic<double_slit::Bohmian>(paths, rng, sig_, d);

  auto vel = [&](double t, std::array<double, 3> X)
  { return double_slit::vel<double_slit::Bohmian>(t, X, sig_, vx, d, m);
  };
  auto stop = [&](double t, std::array<double, 3> X)
  { return X[0] >= dist;
  };

  #pragma omp parallel for
  for (size_t i = 0; i < n; i++)
  { size_t iter = Cash_Karp_45::solve<3>(*paths[i], vel, stop, abs_tol, max_iter);
    if (print_progress)
    { int n_length = std::to_string(n).length();
      printf("%*zu/%zu: %zu\n", n_length, i+1, n, iter);
    }
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<4, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

PyObject *Pauli
( PyObject *sig, double d, PyObject *s, double vx, double dist, double m, size_t n, uint64_t seed, double abs_tol,
  size_t max_iter, bool print_progress
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto sig_ = py_util::from_float_list<3>(sig);
  auto s_ = py_util::from_float_list<3>(s);
  PyGILState_Release(gil);
  std::vector<std::vector<double> *> paths(n);
  for (size_t i = 0; i < n; i++)
    paths[i] = new std::vector<double>;
  pcg_xsl_rr_128_64::State rng {seed, inc_seed};
  double_slit::gen_ic<double_slit::Pauli>(paths, rng, sig_, d);

  auto vel = [&](double t, std::array<double, 3> X)
  { return double_slit::vel<double_slit::Pauli>(t, X, sig_, vx, d, m, s_);
  };
  auto stop = [&](double t, std::array<double, 3> X)
  { return X[0] >= dist;
  };

  #pragma omp parallel for
  for (size_t i = 0; i < n; i++)
  { size_t iter = Cash_Karp_45::solve<3>(*paths[i], vel, stop, abs_tol, max_iter);
    if (print_progress)
    { int n_length = std::to_string(n).length();
      printf("%*zu/%zu: %zu\n", n_length, i+1, n, iter);
    }
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<4, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

PyObject *zigzag
( PyObject *sig, double d, PyObject *s, double vx, double dist, double m, size_t n, uint64_t seed, double abs_tol,
  double p_tol, size_t max_iter, bool print_progress
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto sig_ = py_util::from_float_list<3>(sig);
  auto s_ = py_util::from_float_list<3>(s);
  PyGILState_Release(gil);
  std::vector<std::vector<double> *> paths(n);
  for (size_t i = 0; i < n; i++)
    paths[i] = new std::vector<double>;
  pcg_xsl_rr_128_64::State rng {seed, inc_seed};
  double_slit::gen_ic<double_slit::zigzag>(paths, rng, sig_, d);

  auto vel = [&](double t, std::array<double, 3> X, int chi)
  { return double_slit::vel<double_slit::zigzag>(t, X, sig_, vx, d, m, s_, chi);
  };
  auto rate = [&](double t, std::array<double, 3> X, int chi)
  { return double_slit::rate<double_slit::zigzag>(t, X, sig_, vx, d, m, s_, chi);
  };
  auto stop = [&](double t, std::array<double, 3> X, int chi)
  { return X[0] >= dist;
  };

  auto n_threads = omp_get_max_threads();
  std::vector<pcg_xsl_rr_128_64::State> rngs;
  for (size_t i = 0; i < n_threads; i++)
    rngs.push_back(pcg_xsl_rr_128_64::State {seed+i, inc_seed+i});
  #pragma omp parallel for num_threads(n_threads)
  for (size_t i = 0; i < n; i++)
  { size_t iter = Cash_Karp_45::solve<3, true>(*paths[i], rngs[omp_get_thread_num()], vel, rate, stop, abs_tol, p_tol,
      max_iter);
    if (print_progress)
    { int n_length = std::to_string(n).length();
      printf("%*zu/%zu: %zu\n", n_length, i+1, n, iter);
    }
  }

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<5, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

}
