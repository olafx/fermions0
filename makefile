CXX = /opt/homebrew/Cellar/llvm/19.1.6/bin/clang++
CXXFLAGS = -std=c++20 -shared -fPIC -O3 -ffast-math -fopenmp

PYTHON_VERSION = 3.12
PYTHON_INCLUDE_DIR = /opt/homebrew/Cellar/python@3.12/3.12.7_1/Frameworks/Python.framework/Versions/3.12/include/python3.12
PYTHON_LIB_DIR = /opt/homebrew/Cellar/python@3.12/3.12.7_1/Frameworks/Python.framework/Versions/3.12/lib
NUMPY_INCLUDE_DIR = /Users/ye/venv/main/lib/python3.12/site-packages/numpy/_core/include
NUMPY_LIB_DIR = /Users/ye/venv/main/lib/python3.12/site-packages/numpy/_core/lib
OMP_INCLUDE_DIR = /opt/homebrew/Cellar/libomp/19.1.3/include
OMP_LIB_DIR = /opt/homebrew/Cellar/libomp/19.1.3/lib

INCLUDE = -I$(PYTHON_INCLUDE_DIR) -I$(NUMPY_INCLUDE_DIR) -I$(OMP_INCLUDE_DIR)
LIB = -L$(PYTHON_LIB_DIR) -L$(NUMPY_LIB_DIR) -L$(OMP_LIB_DIR)

LIBS = Lorenz double_slit Stern_Gerlach H_Sch

.PHONY: all clean $(LIBS)
all: $(LIBS)
clean:
	rm -f $(addprefix out/, $(LIBS:=.so))

%.so: src/%.cpp
	$(CXX) $(CXXFLAGS) -o out/$@ $< $(INCLUDE) $(LIB) -lpython$(PYTHON_VERSION) -lnpymath

$(LIBS): %: %.so
