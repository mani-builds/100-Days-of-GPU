#include <torch/extension.h>

// Declare the function
void tiledMatMul_launcher();

// C++ func we call from Python
void tiledMatMul_binding() {
  tiledMatMul_launcher();
}

PYBIND11_MODULE(example_kernels, m) {
  m.def(
    "tiled_matmul", // Name of the Python function to create
    &tiledMatMul_binding, // Corresponding C++ function to call
    "Launches the tiledMatMul kernel" // Docstring
  );
}












