#include <torch/torch.h>
#include <cGodotSharedInterface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2Coords_forward", &Angles2Coords_forward, "Angles2Coords forward");
	m.def("Angles2Coords_backward", &Angles2Coords_backward, "Angles2Coords backward");
}