#include <torch/extension.h>
#include <string>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocator;
typedef std::vector<int, ShmemAllocator> IntVector;
typedef std::vector<float, ShmemAllocator> FloatVector;

class cSharedMemoryTensor{

    private:

        std::string *segment_name = NULL;
        managed_shared_memory *segment = NULL;
        shared_memory_object *object = NULL;

    public:
        
        cSharedMemoryTensor(const std::string &name);
        ~cSharedMemoryTensor();

        void send(const std::string &name, torch::Tensor T);
        torch::Tensor receive(const std::string &name);
};

class cSharedMemorySemaphore{
    private:
        named_semaphore *sem;
        std::string *name;
    public:
        cSharedMemorySemaphore(const std::string &sem_name, int init_count){
            this->name = new std::string(sem_name);
            sem = new named_semaphore(create_only, name->c_str(), init_count);
        };
        ~cSharedMemorySemaphore(){
            sem->remove(name->c_str());
            delete sem;
            delete name;
        };
        void post(){
            sem->post();
        };
        void wait(){
            sem->wait();
        };
};

#define ERROR(x) AT_ASSERTM(true, #x)
#define CHECK_CPU(x) AT_ASSERTM(!(x.type().is_cuda()), #x "must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x,y) AT_ASSERTM(x.dtype()==y, #x " wrong tensor type")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT_TYPE(x, y) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, y)