/* cSharedMemory.cpp */

#include "cSharedMemory.h"

#include <core/os/os.h>


#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <platform/x11_shared/x11_shared.h>

using namespace boost::interprocess;

typedef std::string MyType;

cSharedMemory::cSharedMemory(){
	//Obtain segment_name value
	X11_shared *os = reinterpret_cast<X11_shared*>(OS::get_singleton());
	segment_name = new std::string(os->get_shared_handle().str());
		
	try{
		segment = new managed_shared_memory(open_only, segment_name->c_str());
	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
		shared_memory_object::remove(segment_name->c_str());
	}
};

cSharedMemory::~cSharedMemory(){
	shared_memory_object::remove(segment_name->c_str());
    delete segment;
    delete segment_name;
};

PoolVector<int> cSharedMemory::getArray(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );

	
	PoolVector<int> data;
	try{
		IntVector *shared_vector = segment->find<IntVector> (s_name.c_str()).first;
		for(int i=0; i<shared_vector->size(); i++){
			data.push_back( (*shared_vector)[i] );
		}
		segment->destroy<IntVector>(s_name.c_str());
	}catch(interprocess_exception &ex){
        std::cout<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
	return data;
}

PoolVector<int> cSharedMemory::getArrayBlocking(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );

	
	PoolVector<int> data;
	while(True){
		try{
			IntVector * shared_vector = segment->find<IntVector> (s_name.c_str()).first;
			
			for(int i=0; i<shared_vector->size(); i++){
				data.push_back( (*shared_vector)[i] );
			}
			segment->destroy<IntVector>(s_name.c_str());
			break;
		}catch(interprocess_exception &ex){
			std::cout<<ex.what()<<std::endl;
		}catch(std::exception &ex){
			std::cout<<ex.what()<<std::endl;
		}
	}
	return data;
}

void cSharedMemory::sendArray(const String &name, const PoolVector<int> &array){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	try{
		const ShmemAllocator alloc_inst (segment->get_segment_manager());
		IntVector *shared_vector = segment->construct<IntVector>(s_name.c_str())(alloc_inst);
		for(int i=0; i<array.size(); i++)
			shared_vector->push_back(array[i]);
	}catch(interprocess_exception &ex){
        std::cout<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
}

void cSharedMemory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("getArray", "str"), &cSharedMemory::getArray);
	ClassDB::bind_method(D_METHOD("getArrayBlocking", "str"), &cSharedMemory::getArray);
	ClassDB::bind_method(D_METHOD("sendArray", "str", "array"), &cSharedMemory::sendArray);
}

void cSharedMemorySemaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("post"), &cSharedMemorySemaphore::post);
	ClassDB::bind_method(D_METHOD("wait"), &cSharedMemorySemaphore::wait);
	ClassDB::bind_method(D_METHOD("init", "str"), &cSharedMemorySemaphore::init);
}
