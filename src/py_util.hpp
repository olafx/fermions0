#pragma once
#include <vector>
#include <span>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py_util
{

// Must call this function before using NumPy functionality.
void np_init
()
{ _import_array();
}

template <typename T>
void deallocator
( PyObject *capsule
)
{ auto *p = PyCapsule_GetPointer(capsule, nullptr);
  if (p)
    delete static_cast<T>(p);
}

// Data must be heap allocated so that Python can take ownership.
template <int np_type, typename c_type, typename T>
PyObject *to_np_array
( std::span<npy_intp> dims, c_type *data, T *owner
)
{ auto *np_array = PyArray_SimpleNewFromData(dims.size(), dims.data(), np_type, data);
// Pass along the ownership of the data to Python, so that the destruction
// of the NumPy array also frees the data.
  auto* capsule = PyCapsule_New(static_cast<void *>(owner), nullptr, deallocator<decltype(owner)>);
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(np_array), capsule);
  return np_array;
}

// Data must be heap allocated so that Python can take ownership.
template <int np_type, typename c_type>
PyObject *to_np_array
( std::vector<c_type> *data, npy_intp n_cols
)
{ std::array<npy_intp, 2> dims {static_cast<npy_intp>(data->size()/n_cols), n_cols};
  return to_np_array<np_type, c_type>(dims, data->data(), data);
}

// Elements of data must be heap allocated so that Python can take ownership.
template <size_t n_cols, int np_type, typename c_type>
PyObject *to_list_of_np_arrays
( std::span<std::vector<c_type> *> data
)
{ PyObject *py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++)
    PyList_SetItem(py_list, i, to_np_array<np_type, c_type>(data[i], n_cols));
  return py_list;
}

template <int np_type, typename c_type>
std::tuple<std::span<npy_intp>, c_type *> from_np_array
( PyObject *np_array
)
{ assert(PyArray_Check(np_array));
  auto *np_array_ = reinterpret_cast<PyArrayObject *>(np_array);
  assert(PyArray_TYPE(np_array_) == np_type);
  std::span<npy_intp> dims {PyArray_SHAPE(np_array_), static_cast<size_t>(PyArray_NDIM(np_array_))};
  auto *data = static_cast<c_type *>(PyArray_DATA(np_array_));
  return {dims, data};
}

std::vector<double> from_float_list
( PyObject *py_list
)
{ assert(PyList_Check(py_list));
  auto size = PyList_Size(py_list);
  std::vector<double> list(size);
  for (size_t i = 0; i < size; i++)
  { auto *item = PyList_GetItem(py_list, i);
    assert(PyFloat_Check(item));
    list[i] = PyFloat_AsDouble(item);
  }
  return list;
}

template <size_t expected_size>
std::array<double, expected_size> from_float_list
( PyObject *py_list
)
{ assert(PyList_Check(py_list));
  assert(PyList_Size(py_list) == expected_size);
  std::array<double, expected_size> list;
  for (size_t i = 0; i < expected_size; i++)
  { auto *item = PyList_GetItem(py_list, i);
    assert(PyFloat_Check(item));
    list[i] = PyFloat_AsDouble(item);
  }
  return list;
}

template <size_t expected_size>
std::array<std::complex<double>, expected_size> from_complex_list
( PyObject *py_list
)
{ assert(PyList_Check(py_list));
  assert(PyList_Size(py_list) == expected_size);
  std::array<std::complex<double>, expected_size> list;
  for (size_t i = 0; i < expected_size; i++)
  { auto *item = PyList_GetItem(py_list, i);
    assert(PyComplex_Check(item));
    auto x = PyComplex_AsCComplex(item);
    list[i] = {x.real, x.imag};
  }
  return list;
}

PyObject *to_tuple
( size_t number, PyObject *object
)
{ auto *number_ = PyLong_FromSize_t(number);
  auto *tuple = PyTuple_Pack(2, number_, object);
  Py_DECREF(number_);
  Py_DECREF(object);
  return tuple;
}

} // py_util
