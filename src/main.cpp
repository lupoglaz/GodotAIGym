#include <torch/torch.h>
#include <cGodotSharedInterface.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers", py::arg("i"), py::arg("j"));
	m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;

	py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
		.def("__repr__",
			[](const Pet &a) {
				return "<example.Pet named '" + a.name + "'>";
			}
		);

	py::class_<cSharedMemoryTensor>(m, "SharedMemory")
		.def(py::init<const std::string &>())
		.def("getHandle", &cSharedMemoryTensor::getHandle);
}