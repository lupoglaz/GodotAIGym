#include <torch/torch.h>
#include <cGodotSharedInterface.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
    
	py::class_<cSharedMemoryTensor>(m, "SharedMemoryTensor")
		.def(py::init<const std::string &>())
		.def("send", &cSharedMemoryTensor::send)
		.def("receive", &cSharedMemoryTensor::receive)
		.def("receiveBlocking", &cSharedMemoryTensor::receiveBlocking);

	py::class_<cSharedMemorySemaphore>(m, "SharedMemorySemaphore")
		.def(py::init<const std::string &, int>())
		.def("post", &cSharedMemorySemaphore::post)
		.def("wait", &cSharedMemorySemaphore::wait);
}