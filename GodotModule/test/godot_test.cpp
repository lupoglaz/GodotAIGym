#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>


typedef int MyType;

int main(int argc, char *argv[])
{
    using namespace boost::interprocess;
   
    //Remove shared memory on construction and destruction
    struct shm_remove
    {
        shm_remove() { shared_memory_object::remove("GodotSharedMemory"); }
        ~shm_remove(){ shared_memory_object::remove("GodotSharedMemory"); }
    } remover;

    //Construct managed shared memory
    managed_shared_memory segment(create_only, "GodotSharedMemory", 65536);

    MyType *instance1 = segment.construct<MyType>
        ("Variable 0")     //name of the object
        (0);            //Same two ctor arguments for all objects

    MyType *instance2 = segment.construct<MyType>
        ("Variable 1")     //name of the object
        (1);            //Same two ctor arguments for all objects

    //Launch child process
    std::string s(argv[1]); s += " --handle GodotSharedMemory";
    if(0 != std::system(s.c_str()))
        return 1;

    //Check child has destroyed all objects
    if(segment.find<MyType>("MyType array").first)
        return 1;
   
    return 0;
}