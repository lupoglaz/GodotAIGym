#include <torch/torch.h>
#include <cGodotSharedInterface.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	py::class_<cPersistentIntTensor>(m, "PersistentIntTensor")
		.def("read", &cPersistentIntTensor::read)
		.def("write", &cPersistentIntTensor::write);
	
	py::class_<cPersistentFloatTensor>(m, "PersistentFloatTensor")
		.def("read", &cPersistentFloatTensor::read)
		.def("write", &cPersistentFloatTensor::write);
    
	py::class_<cSharedMemory>(m, "SharedMemory")
		.def(py::init<const std::string &>())
		.def("newIntTensor", &cSharedMemory::newIntTensor)
		.def("newFloatTensor", &cSharedMemory::newFloatTensor);

	py::class_<cSharedMemorySemaphore>(m, "SharedMemorySemaphore")
		.def(py::init<const std::string &, int>())
		.def("post", &cSharedMemorySemaphore::post)
		.def("wait", &cSharedMemorySemaphore::wait)
		.def("timed_wait", &cSharedMemorySemaphore::timed_wait);
}
