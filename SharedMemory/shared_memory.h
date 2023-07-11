/* cSharedMemory.h */
#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <godot_cpp/variant/builtin_types.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/class_db.hpp>
using namespace godot;

#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <istream>
#include <streambuf>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/exception/all.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocatorInt;
typedef allocator<float, managed_shared_memory::segment_manager>  ShmemAllocatorFloat;
typedef std::vector<int, ShmemAllocatorInt> IntVector;
typedef std::vector<float, ShmemAllocatorFloat> FloatVector;

class cPersistentIntTensor : public RefCounted{
    GDCLASS(cPersistentIntTensor, RefCounted);

	private:
		IntVector *vector = NULL;
		int size;

    protected:
        static void _bind_methods();

	public:
        cPersistentIntTensor(){};
		cPersistentIntTensor(IntVector *_vector){
			vector = _vector;
			size = _vector->size();
		}
		~cPersistentIntTensor(){}
		void write(const TypedArray<int> &array){
            //print_line(String("Write int vector:"+String(String::num_int64(size))));
			for(int i=0; i<size; i++)
				(*vector)[i] = array[i];
		};
		TypedArray<int> read(){
            //print_line(String("Read int vector:"+String(String::num_int64(size))));
            TypedArray<int> data;
			for(int i=0; i<size; i++)
				data.push_back( (*vector)[i] );
			return data;
		}
		
};
class cPersistentFloatTensor : public RefCounted {
    GDCLASS(cPersistentFloatTensor, RefCounted);

	private:
		FloatVector *vector = NULL;
		int size;

    protected:
        static void _bind_methods();

	public:
        cPersistentFloatTensor(){};
		cPersistentFloatTensor(FloatVector *_vector){
			vector = _vector;
			size = _vector->size();
		}
		~cPersistentFloatTensor(){}
		void write(const TypedArray<float> &array){
            // print_line(String("Write float vector:"+String(String::num_int64(size))));
			for(int i=0; i<size; i++)
				(*vector)[i] = array[i];
		}
		TypedArray<float> read(){
            //print_line(String("Read float vector:"+String(String::num_int64(size))));
			TypedArray<float> data;
			for(int i=0; i<size; i++)
				data.push_back( (*vector)[i] );
			return data;
		}
};

class cSharedMemory : public RefCounted {
    GDCLASS(cSharedMemory, RefCounted);

private:
    String segment_name;
    managed_shared_memory *segment = NULL;
    bool found;

protected:
    static void _bind_methods();

public:
    cSharedMemory();
    ~cSharedMemory();
    bool init(const String& segment_name);
    Ref<cPersistentFloatTensor> findFloatTensor(const String &name);
    Ref<cPersistentIntTensor> findIntTensor(const String &name);
    bool exists();
};


class cSharedMemorySemaphore : public RefCounted {
    GDCLASS(cSharedMemorySemaphore, RefCounted);
    private:
        String name;
        mapped_region *region;
        interprocess_semaphore *mutex;
    
    protected:
        static void _bind_methods();
    
    public:
        cSharedMemorySemaphore(){;};
        ~cSharedMemorySemaphore(){
            //shared_memory_object::remove(name->c_str());
            delete region;
            delete mutex;
        };
        void init(const String &sem_name){
            name = sem_name;
            std::cout<<"Constructing semaphore "<<std::endl;//<<*name<<std::endl;
            try{
                shared_memory_object object(open_only, (const char*)sem_name.ptr(), read_write);
                region = new mapped_region(object, read_write);
            }catch(interprocess_exception &e){
                //print_line(String("cSharedMemorySemaphore:init:")+String((*name).c_str())+String(":")+String(boost::diagnostic_information(e).c_str()));
                //shared_memory_object::remove(name->c_str());
            }
            std::cout<<"Constructed semaphore "<<std::endl;//<<*name<<std::endl;
        };
        void post(){
            std::cout<<"Post semaphore "<<std::endl;//<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->post();
            std::cout<<"Posted semaphore "<<std::endl;//<<*name<<std::endl;
        };
        void wait(){
            std::cout<<"Wait semaphore "<<std::endl;//<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->wait();
            std::cout<<"Waited semaphore "<<std::endl;//<<*name<<std::endl;
        };
};


#endif