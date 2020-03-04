#include <torch/extension.h>
#include <string>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocatorInt;
typedef allocator<float, managed_shared_memory::segment_manager>  ShmemAllocatorFloat;
typedef std::vector<int, ShmemAllocatorInt> IntVector;
typedef std::vector<float, ShmemAllocatorFloat> FloatVector;

struct TensorDescription{
    std::string type;       //Tensor scalar type
    TensorDescription(const std::string &t_type):type(t_type){}
};

class cSharedMemoryTensor{

    private:

        std::string *segment_name = NULL;
        managed_shared_memory *segment = NULL;
        shared_memory_object *object = NULL;

    public:
        
        cSharedMemoryTensor(const std::string &name);
        ~cSharedMemoryTensor();

        void sendInt(const std::string &name, torch::Tensor T);
        void sendFloat(const std::string &name, torch::Tensor T);
        torch::Tensor receiveInt(const std::string &name);
        torch::Tensor receiveFloat(const std::string &name);
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
};


#define ERROR(x) AT_ASSERTM(true, #x)
#define CHECK_CPU(x) AT_ASSERTM(!(x.type().is_cuda()), #x "must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x,y) AT_ASSERTM(x.dtype()==y, #x " wrong tensor type")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT_TYPE(x, y) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, y)
