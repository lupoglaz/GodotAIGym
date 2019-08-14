#include "cGodotSharedInterface.h"
#include <sstream>

int add(int i, int j) {
    return i + j;
}
cSharedMemoryTensor::cSharedMemoryTensor(const std::string &name){
    segment_name = new std::string(name);

    //Create a managed shared memory segment
    segment = new managed_shared_memory(create_only, segment_name->c_str(), 65536);

    //Allocate a portion of the segment (raw memory)
    managed_shared_memory::size_type free_memory = segment->get_free_memory();
    shptr = segment->allocate(1024/*bytes to allocate*/);

    //Check invariant
    if(free_memory <= segment->get_free_memory())
        throw "Memory corruption";
    std::cout<<"Created segment"<<std::endl;
}
cSharedMemoryTensor::~cSharedMemoryTensor(){
    shared_memory_object::remove(segment_name->c_str());
    delete segment;
    delete segment_name;
    std::cout<<"Removed segment"<<std::endl;
}

std::string cSharedMemoryTensor::getHandle() const{
    managed_shared_memory::handle_t handle = segment->get_handle_from_address(shptr);
    std::stringstream s;
    s << handle << std::ends;
    std::cout<<"Returned handle"<<std::endl;
    return s.str();
}