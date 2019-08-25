/* cSharedMemory.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"
#include "core/pool_vector.h"

#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/exception/all.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocator;
typedef std::vector<int, ShmemAllocator> IntVector;
typedef std::vector<float, ShmemAllocator> FloatVector;


class cSharedMemory : public Reference {
    GDCLASS(cSharedMemory, Reference);

private:
    std::string *segment_name = NULL;
    managed_shared_memory *segment = NULL;
    shared_memory_object *object = NULL;

protected:
    static void _bind_methods();

public:
    cSharedMemory();
    ~cSharedMemory();

    PoolVector<int> getArray(const String &name);
    void sendArray(const String &name, const PoolVector<int> &array);
};


class cSharedMemorySemaphore : public Reference {
    GDCLASS(cSharedMemorySemaphore, Reference);
    private:
        std::string *name;
        mapped_region *region;
        interprocess_semaphore *mutex;
    
    protected:
        static void _bind_methods();
    
    public:
        cSharedMemorySemaphore(){;};
        ~cSharedMemorySemaphore(){
            shared_memory_object::remove(name->c_str());
            delete region;
            delete name;
        };
        void init(const String &sem_name){
            
            std::wstring ws = sem_name.c_str();
	        std::string s_name( ws.begin(), ws.end() );
            name = new std::string(s_name);
            // std::cout<<"Constructing semaphore "<<*name<<std::endl;
            try{
                shared_memory_object object(open_only, name->c_str(), read_write);
                region = new mapped_region(object, read_write);
            }catch(interprocess_exception &e){
                std::cout<<boost::diagnostic_information(e)<<std::endl;
                shared_memory_object::remove(name->c_str());
            }
            // std::cout<<"Constructed semaphore "<<*name<<std::endl;
        };
        void post(){
            // std::cout<<"Post semaphore "<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->post();
            // std::cout<<"Posted semaphore "<<*name<<std::endl;
        };
        void wait(){
            // std::cout<<"Wait semaphore "<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->wait();
            // std::cout<<"Waited semaphore "<<*name<<std::endl;
        };
};

#endif