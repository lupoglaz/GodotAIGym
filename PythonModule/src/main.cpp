#include <torch/torch.h>
#include <cGodotSharedInterface.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
    
	py::class_<cSharedMemoryTensor>(m, "SharedMemoryTensor")
		.def(py::init<const std::string &>())
		.def("sendInt", &cSharedMemoryTensor::sendInt)
		.def("sendFloat", &cSharedMemoryTensor::sendFloat)
		.def("receiveInt", &cSharedMemoryTensor::receiveInt)
		.def("receiveFloat", &cSharedMemoryTensor::receiveFloat);

	py::class_<cSharedMemorySemaphore>(m, "SharedMemorySemaphore")
		.def(py::init<const std::string &, int>())
		.def("post", &cSharedMemorySemaphore::post)
		.def("wait", &cSharedMemorySemaphore::wait);
}