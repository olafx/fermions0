#include <cstdio>
#include <string>
#include "py_util.hpp"
#include "integrate.hpp"

extern "C"
{

PyObject *Lorenz
( double rho, double sigma, double beta, PyObject *X0, double tf, double abs_tol
)
{ auto gil = PyGILState_Ensure();
  py_util::np_init();
  auto [X0_dims, X0_data] = py_util::from_np_array<NPY_DOUBLE, double>(X0);
  PyGILState_Release(gil);
  std::vector<std::vector<double> *> paths(X0_dims[0]);
  for (size_t i = 0; i < paths.size(); i++)
    paths[i] = new std::vector<double>{0, X0_data[3*i  ],
                                          X0_data[3*i+1],
                                          X0_data[3*i+2]};

  auto F = [&](double t, std::array<double, 3> X)
  { return std::array
    { sigma*(X[1]-X[0]),
      X[0]*(rho-X[2])-X[1],
      X[0]*X[1]-beta*X[2]
    };
  };
  auto stop = [&](double t, std::array<double, 3> X)
  { return t >= tf;
  };

  #pragma omp parallel for
  for (size_t i = 0; i < paths.size(); i++)
    Cash_Karp_45::solve<3>(*paths[i], F, stop, abs_tol);

  gil = PyGILState_Ensure();
  auto *list = py_util::to_list_of_np_arrays<4, NPY_DOUBLE, double>(paths);
  PyGILState_Release(gil);
  return list;
}

}
