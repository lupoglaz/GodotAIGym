#include <torch/extension.h>
#include <string>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/exception/all.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocatorInt;
typedef allocator<float, managed_shared_memory::segment_manager>  ShmemAllocatorFloat;
typedef allocator<uint8_t, managed_shared_memory::segment_manager>  ShmemAllocatorUint;
typedef std::vector<int, ShmemAllocatorInt> IntVector;
typedef std::vector<float, ShmemAllocatorFloat> FloatVector;
typedef std::vector<uint8_t, ShmemAllocatorUint> UintVector;

class cPersistentIntTensor{
	private:
		managed_shared_memory *segment = NULL;
		IntVector *vector = NULL;
		int size;
		std::string *name = NULL;
	public:
		cPersistentIntTensor(IntVector *_vector, const std::string &_name, managed_shared_memory *_segment){
			vector = _vector;
			size = vector->size();
			segment = _segment;
			name = new std::string(_name);
		}
		~cPersistentIntTensor(){
			segment->destroy<IntVector>(name->c_str());
			delete name;
		}
		void write(torch::Tensor T){
			for(int i=0; i<size; i++)
				(*vector)[i] = T.data_ptr<int>()[i];
		};
		torch::Tensor read(){
			torch::Tensor T = torch::zeros(size, torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
			for(int i=0; i<size; i++)
				T[i] = (*vector)[i];
			return T;
		}
		
};
class cPersistentFloatTensor{
	private:
		managed_shared_memory *segment = NULL;
		FloatVector *vector = NULL;
		int size;
		std::string *name = NULL;
	public:
		cPersistentFloatTensor(FloatVector *_vector, const std::string &_name, managed_shared_memory *_segment){
			vector = _vector;
			size = vector->size();
			segment = _segment;
			name = new std::string(_name);
		}
		~cPersistentFloatTensor(){
			segment->destroy<FloatVector>(name->c_str());
			delete name;
		}
		void write(torch::Tensor T){
			for(int i=0; i<size; i++)
				(*vector)[i] = T.data_ptr<float>()[i];
		}
		torch::Tensor read(){
			torch::Tensor T = torch::zeros(size, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
			for(int i=0; i<size; i++)
				T[i] = (*vector)[i];
			return T;
		}
};
class cPersistentUintTensor{
	private:
		managed_shared_memory *segment = NULL;
		UintVector *vector = NULL;
		int size;
		std::string *name = NULL;
	public:
		cPersistentUintTensor(UintVector *_vector, const std::string &_name, managed_shared_memory *_segment) {
			vector = _vector;
			size = vector->size();
			segment = _segment;
			name = new std::string(_name);
		}
		~cPersistentUintTensor(){
			segment->destroy<UintVector>(name->c_str());
			delete name;
		}
		void write(torch::Tensor T){
			for(int i = 0; i < size; i++)
				(*vector)[i] = T.data_ptr<int>()[i];
		};
		torch::Tensor read(){
			torch::Tensor T = torch::zeros(size, torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
			for(int i = 0; i < size; i++)
				T[i] = (*vector)[i];
			return T;
		}
};

class cSharedMemoryTensor{

	private:

		std::string *segment_name = NULL;
		managed_shared_memory *segment = NULL;
	public:
		
		cSharedMemoryTensor(const std::string &name);
		~cSharedMemoryTensor();
		
		cPersistentIntTensor* newIntTensor(const std::string &name, int size);
		cPersistentFloatTensor* newFloatTensor(const std::string &name, int size);
		cPersistentUintTensor* newUintTensor(const std::string &name, int size);
};

class cSharedMemorySemaphore{
	private:
		std::string *name;
		mapped_region *region;
		interprocess_semaphore *mutex;
	public:
		cSharedMemorySemaphore(const std::string &sem_name, int init_count);
		~cSharedMemorySemaphore();
		void post();
		void wait();
		void timed_wait(int time);
};


#define ERROR(x) AT_ASSERTM(true, #x)
#define CHECK_CPU(x) AT_ASSERTM(!(x.type().is_cuda()), #x "must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x,y) AT_ASSERTM(x.dtype()==y, #x " wrong tensor type")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT_TYPE(x, y) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, y)
