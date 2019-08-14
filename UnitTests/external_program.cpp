#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib> //std::system
#include <sstream>
#include <iostream>

int main (int argc, char *argv[])
{
    const char *segment_name, *handle_value;
    if(argc<3){
        std::cout<<"Segment name not found"<<std::endl;
        return -1;
    }else if(argc==3){
        segment_name = argv[1];
        handle_value = argv[2];
    }
    std::cout<<"Segment name = "<<segment_name<<handle_value<<std::endl;
    using namespace boost::interprocess;
    //Open managed segment
    managed_shared_memory segment(open_only, segment_name);
    std::cout<<"Opened segment"<<std::endl;
    //An handle from the base address can identify any byte of the shared
    //memory segment even if it is mapped in different base addresses
    managed_shared_memory::handle_t handle = 0;

    //Obtain handle value
    std::stringstream s; s << handle_value; s >> handle;
    std::cout<<"Got handle value"<<std::endl;

    //Get buffer local address from handle
    void *msg = segment.get_address_from_handle(handle);

    //Deallocate previously allocated memory
    segment.deallocate(msg);
    std::cout<<"Finished"<<std::endl;
   return 0;
}