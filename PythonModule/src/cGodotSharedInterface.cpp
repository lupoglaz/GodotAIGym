#include "cGodotSharedInterface.h"
#include <sstream>

cSharedMemoryTensor::cSharedMemoryTensor(const std::string &name){
    segment_name = new std::string(name);
    try{
        //Create a managed shared memory segment
        segment = new managed_shared_memory(create_only, segment_name->c_str(), 65536);
    }catch (interprocess_exception& e) {
        shared_memory_object::remove(segment_name->c_str());
    }
}
cSharedMemoryTensor::~cSharedMemoryTensor(){
    shared_memory_object::remove(segment_name->c_str());
    delete segment;
    delete segment_name;
}

void cSharedMemoryTensor::sendInt(const std::string &name, torch::Tensor T){
    CHECK_CPU_INPUT_TYPE(T, torch::kInt);
    if(T.ndimension() != 1){
        ERROR("Number of dimensions should be 1");
    }
    if(segment->find<IntVector>(name.c_str()).first){
        ERROR("Variable already exists");
    }
    try{
        const ShmemAllocatorInt alloc_inst (segment->get_segment_manager());
        IntVector *myvector = segment->construct<IntVector>(name.c_str())(alloc_inst);
        myvector->resize(T.size(0));
        myvector->assign(T.data<int>(), T.data<int>() + T.size(0));
    }catch(interprocess_exception &ex){
        std::cout<<name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
}

void cSharedMemoryTensor::sendFloat(const std::string &name, torch::Tensor T){
    CHECK_CPU_INPUT_TYPE(T, torch::kFloat);
    if(T.ndimension() != 1){
        ERROR("Number of dimensions should be 1");
    }
    if(segment->find<FloatVector>(name.c_str()).first){
        ERROR("Variable already exists");
    }
    try{
        const ShmemAllocatorFloat alloc_inst (segment->get_segment_manager());
        FloatVector *myvector = segment->construct<FloatVector>(name.c_str())(alloc_inst);
        myvector->resize(T.size(0));
        myvector->assign(T.data<float>(), T.data<float>() + T.size(0));
    }catch(interprocess_exception &ex){
        std::cout<<name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
}

torch::Tensor cSharedMemoryTensor::receiveInt(const std::string &name){
    torch::Tensor T;
    try{
        IntVector *myvector = segment->find<IntVector> (name.c_str()).first;
        T = torch::zeros(myvector->size(), torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
        for(int i=0; i<myvector->size(); i++){
            T[i] = (*myvector)[i];
        }
        segment->destroy<IntVector>(name.c_str());
    }catch(interprocess_exception &ex){
        std::cout<<name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
    return T;
}

torch::Tensor cSharedMemoryTensor::receiveFloat(const std::string &name){
    torch::Tensor T;
    try{
        FloatVector *myvector = segment->find<FloatVector> (name.c_str()).first;
        T = torch::zeros(myvector->size(), torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
        for(int i=0; i<myvector->size(); i++){
            T[i] = (*myvector)[i];
        }
        segment->destroy<FloatVector>(name.c_str());
    }catch(interprocess_exception &ex){
        std::cout<<name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
    return T;
}

cSharedMemorySemaphore::cSharedMemorySemaphore(const std::string &sem_name, int init_count){
    try{
        name = new std::string(sem_name);
        shared_memory_object object(open_or_create, name->c_str(), read_write);
        object.truncate(sizeof(interprocess_semaphore));
        region = new mapped_region(object, read_write);
        void *addr = region->get_address();
        mutex = new (addr) interprocess_semaphore(init_count);
    }catch(interprocess_exception &ex){
        std::cout<<sem_name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }

}

cSharedMemorySemaphore::~cSharedMemorySemaphore(){
    shared_memory_object::remove(name->c_str());
    delete region;
    delete name;
}

void cSharedMemorySemaphore::post(){
    // std::cout<<"Post semaphore "<<*name<<std::endl;
    try{
        //Important: if the mutex is not cast here, it will give segfault
        mutex = static_cast<interprocess_semaphore*>(region->get_address());
        mutex->post();
    }catch(boost::interprocess::interprocess_exception &ex){
        std::cout<<*name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
    // std::cout<<"Posted semaphore "<<*name<<std::endl;
}
void cSharedMemorySemaphore::wait(){
    // std::cout<<"wait semaphore "<<*name<<std::endl;
    try{
        mutex = static_cast<interprocess_semaphore*>(region->get_address());
        mutex->wait();
    }catch(boost::interprocess::interprocess_exception &ex){
        std::cout<<*name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
    // std::cout<<"wait semaphore "<<*name<<std::endl;
}
