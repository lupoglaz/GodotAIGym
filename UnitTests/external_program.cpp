#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <cstdlib> //std::system
#include <sstream>
#include <iostream>
#include <vector>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocator;
typedef std::vector<int, ShmemAllocator> IntVector;
typedef std::vector<float, ShmemAllocator> FloatVector;

int main (int argc, char *argv[])
{
    const char *segment_name, *handle_name_read, *handle_name_write;
    if(argc<4){
        std::cout<<"Segment name not found"<<std::endl;
        return -1;
    }else if(argc==4){
        segment_name = argv[1];
        handle_name_read = argv[2];
        handle_name_write = argv[3];
    }
    // std::cout<<"Segment name = "<<segment_name<<" handle name to read "<<handle_name_read<<" handle name to wite "<<handle_name_write<<std::endl;
    
    try {
        //Open managed shared memory
        managed_shared_memory segment(open_only, segment_name);
        
        IntVector *v = segment.find<IntVector> (handle_name_read).first;
        std::cout<<"C++ from Python = [";
        for(int i=0;i<v->size(); i++){
            std::cout<<(*v)[i]<<", ";
        }
        std::cout<<"]";
        segment.destroy<IntVector>(handle_name_read);

        const ShmemAllocator alloc_inst (segment.get_segment_manager());
        IntVector *myvector = segment.construct<IntVector>(handle_name_write)(alloc_inst);
        for(int i=0; i<10; i++){
            myvector->push_back(i);
        }

        return 0;
    } catch (interprocess_exception& e) {
        std::cout << e.what( ) << std::endl;
        return 1;
    }
}