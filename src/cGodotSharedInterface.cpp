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

void cSharedMemoryTensor::send(const std::string &name, torch::Tensor T){
    CHECK_CPU_INPUT_TYPE(T, torch::kInt);
    if(T.ndimension() != 1){
        ERROR("Number of dimensions should be 1");
    }
    if(segment->find<IntVector>(name.c_str()).first){
        ERROR("Variable already exists");
    }

    const ShmemAllocator alloc_inst (segment->get_segment_manager());
    IntVector *myvector = segment->construct<IntVector>(name.c_str())(alloc_inst);
    myvector->resize(T.size(0));
    myvector->assign(T.data<int>(), T.data<int>() + T.size(0));
    
}

torch::Tensor cSharedMemoryTensor::receive(const std::string &name){
    IntVector *myvector = segment->find<IntVector> (name.c_str()).first;
    torch::Tensor T = torch::zeros(myvector->size(), torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
    
    for(int i=0; i<myvector->size(); i++){
        T[i] = (*myvector)[i];
    }

    segment->destroy<IntVector>(name.c_str());
    return T;
}
