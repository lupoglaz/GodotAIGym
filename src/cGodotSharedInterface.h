#include <torch/extension.h>
#include <string>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

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
        std::string *name;
        boost::interprocess::interprocess_semaphore *mutex;
    public:
        cSharedMemorySemaphore(const std::string &sem_name, int init_count){
            name = new std::string(sem_name);
            shared_memory_object object(create_only, name->c_str(), read_write);
            object.truncate(sizeof(boost::interprocess::interprocess_semaphore));
            mapped_region region( object, read_write);
            void * addr = region.get_address();

            //Construct the shared structure in memory
            mutex = new (addr) boost::interprocess::interprocess_semaphore(init_count);

        };
        ~cSharedMemorySemaphore(){
            shared_memory_object::remove(name->c_str());
            delete name;
            delete mutex;
        };
        void post(){
            mutex->post();
        };
        void wait(){
            mutex->wait();
        };
};


#define ERROR(x) AT_ASSERTM(true, #x)
#define CHECK_CPU(x) AT_ASSERTM(!(x.type().is_cuda()), #x "must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x,y) AT_ASSERTM(x.dtype()==y, #x " wrong tensor type")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT_TYPE(x, y) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, y)